addpath(genpath('~/ADMMSoftmaxCode'))

clear all; 
% clc;

N = 50000; Nval = 0.2*N;
[Dtrain,Ctrain,Dtest,Ctest] = setupMNIST(N, Nval);

Dtrain = reshape(Dtrain, 28*28, N);
Dtest = reshape(Dtest, 28*28, Nval); 

fprintf('maxY = %1.2e, minY = %1.2e', max(Dtrain(:)), min(Dtrain(:)));

channelsIn = 1; 
channelsOut = 9;
nImg = [28 28];

fprintf(' number of training examples: %d \n\n', N);


%% extreme learning
% kernel size
sK = [3, 3, channelsIn, channelsOut]; %3x3 convolution window
Ker = convFFT(nImg, sK);
th   = randn(nTheta(Ker),1);
% load('K.mat')
K  = getOp(Ker,th);
Dtrain = tanh(K*Dtrain); Dtest = tanh(K*Dtest);

Dtrain = reshape(Dtrain, [], N); Dtest = reshape(Dtest, [], Nval);


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
    
alpha = 1e-1;
fprintf('\n\n ALPHA = %1.2e \n', alpha);
    
%% set up optimization
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fTest   = classObjFctn(pLoss,pRegW,Dtest,Ctest);

f.pLoss.addBias=addBias; fTest.pLoss.addBias=addBias;

vec     = @(x) x(:);

if addBias==false
    W0      = vec(randn(nf,nc));
else
    W0      = vec(randn(nf+1,nc));
end

% newton setup
opt      = newton('out',1);
opt.out  = 2;
opt.atol = 1e-12;
opt.rtol = 1e-12;
opt.maxIter= 1000;
opt.LS.maxIter=20;
opt.linSol.maxIter=20;
opt.linSol.tol=1e-2;
opt.stoppingTime = 500;
    
%% solve
tSolve = tic
[Wopt, hisOpt] = solve(opt,f,W0, fTest);
tSolve = toc(tSolve)

saveResults = 0;
if saveResults==1
    save('newtonResultsMNIST.mat', 'hisOpt', 'Wopt', 'alpha')
end