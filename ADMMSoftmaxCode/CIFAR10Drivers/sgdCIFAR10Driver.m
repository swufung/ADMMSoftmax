clear all; 

addpath(genpath('/home/swu69/ADMMSoftmaxCode/')); % euler

layer = 'pool5'; N      = 50000; Nval = 0.2*N;
[Dtrain,Ctrain,Dval,Cval] = setupCIFAR10AlexNet(N, Nval, layer);
Dtrain = double(Dtrain); Dval = double(Dval);
nf = size(Dtrain,1) 
nc = size(Ctrain,1) 

fprintf('maxDtrain = %1.2e, minDtrain = %1.2e\n', max(Dtrain(:)), min(Dtrain(:)));
fprintf('maxDval = %1.2e, minDval = %1.2e\n', max(Dval(:)), min(Dval(:)));

Dtrain    = normalizeData(Dtrain, size(Dtrain,1));
Dval      = normalizeData(Dval, size(Dval,1));
fprintf('maxDtrain = %1.2e, minDtrain = %1.2e\n', max(Dtrain(:)), min(Dtrain(:)));
fprintf('maxDval = %1.2e, minDval = %1.2e\n', max(Dval(:)), min(Dval(:)));

%% regularization

alpha = 1;

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


lr0 = 0.1;
alpha = 1; miniBatch=30;

fprintf('\n\n lr0 = %1.2e, ALPHA = %1.2e, minibatch = %d \n', lr0, alpha, miniBatch);

%% set up optimization
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fTest   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=addBias; fTest.pLoss.addBias=addBias;

vec     = @(x) x(:);

if addBias==false
    w      = vec(zeros(nf,nc));
else
    W0      = vec(zeros(nf+1,nc));
end

%% sgd setup
opt           = sgd('out',1);
opt.nesterov=false;
opt.rtol      = 1e-3;
opt.atol      = 1e-3;
opt.miniBatch = miniBatch;
opt.maxEpochs = 100;
% opt.learningRate = @(epoch) lr0/sqrt(epoch);
opt.learningRate = lr0;
opt.stoppingTime = 500; %seconds


%% solve
tSolve = tic
[wOpt, hisOpt] = solve(opt,f,W0, fTest);
tSolve = toc(tSolve)

fprintf('\n\n ALPHA = %1.2e, fTrain = %1.2e, fTest = %1.2e, trainAcc = %1.2f, testAcc=%1.2f\n\n', ...
    alpha, hisOpt.his(end,9), hisOpt.his(end,13), hisOpt.his(end,10), hisOpt.his(end,13)); %14 for newton

atol = opt.atol; rtol = opt.rtol; 
miniBatch = opt.miniBatch;
save('sgdResultsCIFAR10.mat', 'hisOpt', 'wOpt', 'alpha', 'atol', 'miniBatch', 'rtol', 'lr0', 'miniBatch')