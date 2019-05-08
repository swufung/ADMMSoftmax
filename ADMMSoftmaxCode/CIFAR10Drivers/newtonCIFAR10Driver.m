clear all; 

addpath(genpath('~/ADMMSoftmaxCode/')); 


layer = 'pool5'; 
N      = 50000; % training examples
Nval = 0.2*N;   % validation examples
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

%% set up optimization
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fVal   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=addBias; fVal.pLoss.addBias=addBias;

vec     = @(x) x(:);

if addBias==false
    W0      = vec(zeros(nf,nc));
else
    W0      = vec(zeros(nf+1,nc));
end

%% newton setup
opt                 = newton('out',1);
opt.out             = 2;
opt.atol            = 1e-12;
opt.rtol            = 1e-12;
opt.maxIter         = 100;
opt.LS.maxIter      = 10;
opt.linSol.maxIter  = 20;
opt.linSol.tol      = 1e-2;
opt.stoppingTime    = 300;


%% solve
tSolve = tic
[Wopt, hisOpt] = solve(opt,f,W0, fVal);
tSolve = toc(tSolve)

% saveResults = 1;
% if saveResults==1
%     save('newtonResultsCIFAR10.mat', 'hisOpt', 'Wopt', 'alpha')
% end
