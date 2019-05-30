clear all; 
addpath(genpath('~ADMMSoftmaxCode'))

N = 50000; Nval = 0.2*N;
[Dtrain,Ctrain,Dval,Cval] = setupMNIST(N, Nval);

Dtrain = reshape(Dtrain, 28*28, N);
Dval = reshape(Dval, 28*28, Nval); 

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
Dtrain = tanh(K*Dtrain); Dval = tanh(K*Dval);

Dtrain = reshape(Dtrain, [], N); Dval = reshape(Dval, [], Nval);


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

% % % L = speye(nf);
% % % if addBias==true
% % %     L = sparse([L zeros(size(L,1),1); zeros(1,size(L,2)) 1]);
% % % end
% % % Lout = sparse(genBlkDiag(L, nc-1));

fprintf('size of Lout = %d...\n', size(Lout,1))
fprintf('length of W = %d...\n', nf*nc)
    
alpha = 1e-8;
fprintf('\n\n ALPHA = %1.2e \n', alpha);
    
%% set up optimization
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fVal   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=addBias; fVal.pLoss.addBias=addBias;

vec     = @(x) x(:);

if addBias==false
    w0 = zeros(nf,nc);
else
    w0 = zeros(nf+1,nc);
end

% newton setup
opt      = newton('out',1);
opt.out  = 2;
opt.atol = 1e-6;
opt.rtol = 1e-6;
opt.maxIter= 100;
opt.LS.maxIter=20;
opt.linSol.maxIter=20;
opt.linSol.tol=1e-2;
opt.stoppingTime = 300;
    
%% train
tSolve = tic
[wOpt, hisOpt] = solve(opt,f,w0(:), fVal);
tSolve = toc(tSolve)

save('newtonResultsMNIST.mat', 'hisOpt', 'wOpt', 'alpha')
