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
K  = getOp(Ker,th);
Dtrain = tanh(K*Dtrain); 
Dval  = tanh(K*Dval);
Dtest = tanh(K*Dtest);

Dtrain = reshape(Dtrain, [], Ntrain); 
Dval   = reshape(Dval, [], Nval); 
Dtest  = reshape(Dtest, [], Ntest);

nf = size(Dtrain,1);

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

%%
% for evaluating misfits accuracies
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLossZ();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fVal   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=addBias; fVal.pLoss.addBias=addBias;


%% initial admm values
rho0 = 1e-7; 
maxIter = 10000; atol = 1e-12; rtol = 1e-14;
his = zeros(maxIter,14); out=1; varRho=0; scaleRho = 2; mu = 10;
rhoLowerBound = 1e-16;
rhoUpperBound = 1e3;
% 1)iter 2)Fw 3)Fval, 4)trainAcc 5)valAcc 6)Ziters 7)Fz 8)Zres 9)lagrangian
%10) resPri 11)epsPri 12)resDual 13)epsDual 14)rho 15)currentRuntime

%% LeastSquares parameters
lsSolver = 'qr';
% lsSolver = 'backslash';

%% Z-step parameters
maxIterZ = 100; % max number of Z newton iters
linSolMaxIterZ = 50; % max number of CG iters per newton step in Z step
lsMaxIterZ= 50; % max number of linesearch armijo iters per lin sol in Z step
atolZ = 1e-8; rtolZ=1e-8;
outZ = 0;
linSolTolZ = 1e-8; % tolerance of linear solver (steihaug CG) for Z newton step
%% stopping criteria
% stoppingCrit{1} = 'regular';
% stoppingCrit{1} = 'training'; stoppingCrit{2} = 90; % stop when 90% training
stoppingCrit{1} = 'runtime'; stoppingCrit{2} = 300; % stop after 10 seconds
% stoppingCrit{1} = 'maxiters'; stoppingCrit{2} = 50; % stop after 50 iters

if addBias==true
    w0      = (zeros(nc,nf+1));
    Dtrain  = [Dtrain; ones(1,Ntrain)];
    Dtest   = [Dtest; ones(1,Ntest)]; 
    Dval    = [Dval; ones(1,Nval)];
    Wref    = zeros(nc,nf+1);
else
    w0      = (zeros(nc,nf));
    Wref    = zeros(nc,nf);
end

%% set up param structure

param.maxIter         = maxIter;
param.stoppingCrit    = stoppingCrit;
param.varRho          = varRho;
param.rhoLowerBound   = rhoLowerBound;
param.rhoUpperBound   = rhoUpperBound;
param.mu              = mu;
param.atol            = atol;
param.rtol            = rtol;
param.alpha           = alpha;
param.lsSolver        = lsSolver;
param.addBias         = addBias;
param.rho0            = rho0;
param.scaleRho        = scaleRho;
param.out             = 1;


param.Wref            = Wref;
param.Dtrain          = Dtrain;
param.Dval            = Dval;
param.Ctrain          = Ctrain;
param.Cval            = Cval;
param.L               = L;

param.f               = f; % class obj func with training data
param.fVal            = fVal; % class obj func with validation data

% z parameters
param.atolZ           = atolZ;
param.rtolZ           = rtolZ; 
param.maxIterZ        = maxIterZ;
param.linSolMaxIterZ  = linSolMaxIterZ;
param.linSolTolZ      = linSolTolZ;
param.lsMaxIterZ      = lsMaxIterZ;
param.outZ            = outZ;


%% train
[wFinal, wOptLoss, wOptAcc, hisOpt] = admmSoftmax(w0, param);
% [wOpt, hisOpt] = admmSoftmaxConditionNumber(w0, param);

%% save best validation loss values
WDLoss = reshape(wOptLoss, nc, [])*Dval;

[FcValLoss, paraValLoss] = fVal.pLoss.getMisfit(WDLoss, Cval);
errValLoss = paraValLoss(3);
accValLoss = 100*(Nval-errValLoss)/Nval;

% save best validation accuracy values
WDAcc = reshape(wOptAcc, nc, [])*Dval;

[FcValAcc, paraValAcc] = fVal.pLoss.getMisfit(WDAcc, Cval);
errValAcc = paraValAcc(3);
accValAcc = 100*(Nval-errValAcc)/Nval;

fprintf('\n ------ VALIDATION RESULTS ------ \n')
fprintf('\n wOptLoss: fVal = %1.2e, accVal = %1.2f\n', FcValLoss, accValLoss);
fprintf('\n wOptAcc: fVal = %1.2e, accVal = %1.2f\n', FcValAcc, accValAcc);


%% check testing accuracy
pLossTest = softmaxLoss();
fTest     = classObjFctn(pLossTest,pRegW,Dtest(1:end-1,:),Ctest);

% weights that minimize validation misfit
[FcTestLoss, paraTestLoss] = fTest.eval(wOptLoss(:));
errTestLoss = paraTestLoss.hisLoss(3);
accTestLoss = 100*(Ntest-errTestLoss)/Ntest;

% weights that maximize validation accuracy
[FcTestAcc, paraTestAcc] = fTest.eval(wOptAcc(:));
errTestAcc = paraTestAcc.hisLoss(3);
accTestAcc = 100*(Ntest-errTestAcc)/Ntest;

fprintf('\n ------ TESTING RESULTS ------ \n')
fprintf('\n wOptLoss: fTest = %1.2e, accTest = %1.2f\n', FcTestLoss, accTestLoss);
fprintf('\n wOptAcc: fTest = %1.2e, accTest = %1.2f\n', FcTestAcc, accTestAcc);


saveResults=0;
if saveResults==1
    save('admmResultsMNIST.mat', 'hisOpt', 'wOptLoss', 'wOptAcc','wFinal', 'K', 'alpha', 'atol', 'rtol', 'atolZ', 'rtolZ', 'linSolMaxIterZ', 'lsMaxIterZ', 'maxIterZ', 'rho0', 'FcTestLoss', 'accTestLoss', 'FcTestAcc', 'accTestAcc')
end