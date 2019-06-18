clear all; 

addpath(genpath('/home/swu69/ADMMSoftmaxCode/')); % euler

N = 50000; Ntrain = 0.8*N; Nval = 0.2*N; Ntest = 0.2*N;

layer = 'pool5';
[Dtrain,Ctrain,Dval,Cval,Dtest,Ctest] = setupCIFAR10AlexNet(N, Ntest, layer);
Dtrain = double(Dtrain); Dval = double(Dval); Dtest = double(Dtest);
nf = size(Dtrain,1); 
nc = size(Ctrain,1);

fprintf('maxDtrain = %1.2e, minDtrain = %1.2e\n', max(Dtrain(:)), min(Dtrain(:)));
fprintf('maxDval = %1.2e, minDval = %1.2e\n', max(Dval(:)), min(Dval(:)));

Dtrain    = normalizeData(Dtrain, size(Dtrain,1));
Dval      = normalizeData(Dval, size(Dval,1));
Dtest     = normalizeData(Dtest, size(Dtest,1));
fprintf('maxDtrain = %1.2e, minDtrain = %1.2e\n', max(Dtrain(:)), min(Dtrain(:)));
fprintf('maxDval = %1.2e, minDval = %1.2e\n', max(Dval(:)), min(Dval(:)));


%% regularization

alpha = 1e-1; 

addBias=true;
% % % % nImg = [27 27]; channelsOut = 96; % pool1
nImg = [6 6]; channelsOut = 256; % pool 5
% smoothness reg. operator
fprintf('using smoothness! reg. operator...\n')
Ltemp = getLaplacian(nImg, 1./nImg);

L = genBlkDiag(Ltemp,channelsOut-1);

%  add bias to laplace operator 
if addBias==true
    L = sparse([L zeros(size(L,1),1); zeros(1,size(L,2)) 1]);
end

Lout = sparse(genBlkDiag(L, nc-1));

% account for mesh size: 
Lout = Lout/(nf);
L    = L/(nf);

fprintf('size of Lout = %d = %1.2e...\n', size(Lout,1))
fprintf(' max of Lout... ');
max(Lout(:))
fprintf('length of W = %d...\n', nf*nc)

%% set up optimization
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fVal   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=addBias; fVal.pLoss.addBias=addBias;

vec     = @(x) x(:);

if addBias==false
    w0      = vec(zeros(nf,nc));
else
    w0      = vec(zeros(nf+1,nc));
end

%% newton setup
% % % opt                 = newtonSoftmax('out',1);
% % % opt.out             = 2;
% % % opt.atol            = 1e-12;
% % % opt.rtol            = 1e-12;
% % % opt.maxIter         = 1000;
% % % opt.LS.maxIter      = 10;
% % % opt.linSol.maxIter  = 20;
% % % opt.linSol.tol      = 1e-2;
% % % opt.stoppingTime    = 300;


opt                 = lbfgsSoftmax('out',1);
opt.maxIter         = 1000;
opt.atol            = 1e-12;
opt.rtol                = 1e-12;
opt.maxStep             = 1;
opt.LS.maxIter      = 10;
opt.linSol.maxIter  = 20;
opt.linSol.tol      = 1e-2;
opt.maxlBFGS        = 10;
opt.stoppingTime    = 300;

%% train
tSolve = tic
[wFinal, wOptLoss, wOptAcc, hisOpt] = solve(opt,f,w0(:), fVal);
tSolve = toc(tSolve)

%% save best validation, and testing loss and accuracy values
% weights that minimize validation misfit on validation data
pLossVal = softmaxLoss();
fVal2    = classObjFctn(pLossVal,pRegW,Dval,Cval);

[FcValLoss, paraValLoss] = fVal2.eval(wOptLoss(:));
errValLoss = paraValLoss.hisLoss(3);
accValLoss = 100*(Nval-errValLoss)/Nval;

% weights that minimize validation misfit on validation data
[FcValAcc, paraValAcc] = fVal2.eval(wOptAcc(:));
errValAcc = paraValAcc.hisLoss(3);
accValAcc = 100*(Nval-errValAcc)/Nval;

fprintf('\n ------ VALIDATION RESULTS ------ \n')
fprintf('\n wOptLoss: fVal = %1.2e, accVal = %1.2f\n', FcValLoss, accValLoss);
fprintf('\n wOptAcc: fVal = %1.2e, accVal = %1.2f\n', FcValAcc, accValAcc);


% testing dataset
pLossTest = softmaxLoss();
fTest     = classObjFctn(pLossTest,pRegW,Dtest,Ctest);

% weights that minimize validation misfit on testing data
[FcTestLoss, paraTestLoss] = fTest.eval(wOptLoss(:));
errTestLoss = paraTestLoss.hisLoss(3);
accTestLoss = 100*(Ntest-errTestLoss)/Ntest;

% weights that minimize validation misfit on testing data
[FcTestAcc, paraTestAcc] = fTest.eval(wOptAcc(:));
errTestAcc = paraTestAcc.hisLoss(3);
accTestAcc = 100*(Ntest-errTestAcc)/Ntest;

fprintf('\n ------ TESTING RESULTS ------ \n')
fprintf('\n wOptLoss: fTest = %1.2e, accTest = %1.2f\n', FcTestLoss, accTestLoss);
fprintf('\n wOptAcc: fTest = %1.2e, accTest = %1.2f\n', FcTestAcc, accTestAcc);



saveResults = 0;
if saveResults==1
    save('lbfgsResultsCIFAR10.mat', 'hisOpt', 'wOptLoss', 'wOptAcc', 'wFinal', 'alpha', 'FcTestAcc', 'accTestAcc', 'FcTestLoss', 'accTestLoss')
end