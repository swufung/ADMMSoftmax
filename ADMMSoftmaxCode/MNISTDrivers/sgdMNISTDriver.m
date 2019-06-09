addpath(genpath('/home/swu69/ADMMSoftmaxCode/')); % euler

clear all; 
Ntrain = 40000; Nval = 10000; N = Ntrain + Nval; Ntest = 0.2*N;
[Dtrain, Ctrain, Dval, Cval, Dtest, Ctest] = setupMNIST2(N,Ntest);

% Dtrain, Ctrain = training data
% Dval, Cval = validation data
% Dtest, Ctest = testing data

Dtrain = reshape(Dtrain, 28*28, Ntrain);
Dval   = reshape(Dval, 28*28, Nval);
Dtest  = reshape(Dtest, 28*28, Ntest); 

fprintf('maxY = %1.2e, minY = %1.2e', max(Dtrain(:)), min(Dtrain(:)));


%% 
channelsIn = 1; 
channelsOut = 9;
nImg = [28 28];

nc = size(Ctrain,1);

%% extreme learning
% kernel size
sK = [3, 3, channelsIn, channelsOut]; %3x3 convolution window
Ker = convFFT(nImg, sK);
th   = randn(nTheta(Ker),1);
% load('K.mat')
K  = getOp(Ker,th);
Dtrain = tanh(K*Dtrain); 
Dval  = tanh(K*Dval);
Dtest = tanh(K*Dtest);

Dtrain = reshape(Dtrain, [], Ntrain); 
Dval  = reshape(Dval, [], Nval);
Dtest = reshape(Dtest, [], Ntest);


ne = N;
nf = size(Dtrain,1); nc = size(Ctrain,1);

%% start optimization
addBias=true;

%% regularization
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

fprintf('size of Lout = %d...\n', size(Lout,1))
fprintf('length of W = %d...\n', nf*nc)
    
alpha = 1e-6;
fprintf('\n\n ALPHA = %1.2e \n', alpha);

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

miniBatch = 300; lr0 = 0.1; 

%% sgd setup
opt           = sgdSoftmax('out',1);
opt.nesterov=false;
opt.rtol      = 1e-3;
opt.atol      = 1e-3;
opt.miniBatch = miniBatch;
opt.maxEpochs = 1000;
opt.learningRate = lr0;
opt.stoppingTime = 300; %seconds


%% train
tSolve = tic
[wFinal, wOptLoss, wOptAcc, hisOpt] = solve(opt,f,w0, fVal);
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


atol = opt.atol; rtol = opt.rtol; miniBatch = opt.miniBatch;

saveResults=0;
if saveResults==1
    save('sgdResultsMNIST.mat', 'hisOpt', 'wOptLoss', 'wOptAcc', 'wFinal', 'alpha', 'atol', 'miniBatch', 'rtol', 'lr0', 'FcTestLoss', 'accTestLoss', 'FcTestAcc', 'accTestAcc')
end