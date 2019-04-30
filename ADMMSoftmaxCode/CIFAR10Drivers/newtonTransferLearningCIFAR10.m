clear all; clc;

addpath(genpath('~/ADMMSoftmaxCode'))

% set up original features
[Dtrain,Ctrain,Dval,Cval] = setupCIFAR10(60000);

ntrain = size(Dtrain,2);
nval = size(Dval,2);


%% propagate features through AlexNet
% load the network trained with imageNet
net = alexnet;
net.Layers
inputSize = net.Layers(1).InputSize;  %227*227*3

% build imageAugmenter and augment the CIFAR10 to fit the pretrained network 
pixelRange = [-4 4];
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandXTranslation',pixelRange, ...
                                     'RandYTranslation',pixelRange);
                                
augimdsTrain = augmentedImageDatastore(inputSize,reshape(Dtrain,[32,32,3,ntrain]),Ctrain');
augimdsVal = augmentedImageDatastore(inputSize,reshape(Cval,[32,32,3,nval]),Cval');

% compute layer activations, ts. output of the layer with inputs

layer = 'fc7';

featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresVal = activations(net,augimdsVal,layer,'OutputAs','rows');


%% 

N = 50000; Nval = 0.2*N;

Dtrain = double(featuresTrain(1:N,:))';
Dval  = double(featuresVal(1:Nval,:))';

Ctrain = double(Ctrain(:,1:N)); Cval = double(Cval(:,1:Nval));

nf = size(Dtrain,1); nc = size(Ctrain,1);

fprintf(' number of training examples: %d \n\n', N);

%% start optimization
addBias=true;

%% regularization

fprintf('maxY = %1.2e, minY = %1.2e', max(Dtrain(:)), min(Dtrain(:)));

L = speye(nf);
if addBias==true
    L = sparse([L zeros(size(L,1),1); zeros(1,size(L,2)) 1]);
end
Lout = sparse(genBlkDiag(L, nc-1));
Lout = Lout/(nf);
L    = L/(nf);

    
alpha = 1;
fprintf('\n\n ALPHA = %1.2e \n', alpha);

%% set up optimization
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fTest   = classObjFctn(pLoss,pRegW,Dval,Ctest);

f.pLoss.addBias=addBias; fTest.pLoss.addBias=addBias;

vec     = @(x) x(:);

if addBias==false
    W0      = vec(randn(nf,nc));
else
    W0      = vec(randn(nf+1,nc));
end

%% newton setup
opt      = newton('out',1);
opt.out  = 2;
opt.atol = 1e-12;
opt.rtol = 1e-12;
opt.maxIter= 5000;
opt.LS.maxIter=50;
opt.linSol.maxIter=50;
opt.linSol.tol=1e-4;
opt.stoppingTime = 500;


%% solve
tSolve = tic
[Wopt, hisOpt] = solve(opt,f,W0, fTest);
tSolve = toc(tSolve)

saveResults = 0;
if saveResults==1
    save('newtonResultsCIFAR10.mat', 'hisOpt', 'Wopt', 'alpha')
end