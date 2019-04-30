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

N = 50000; Nval = 0.2*N;

Dtrain = double(featuresTrain(1:N,:))';
Dval  = double(featuresTest(1:Nval,:))';

Ctrain = double(Ctrain(:,1:N)); Cval = double(Ctest(:,1:Nval));

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


lr0 = 10;
alpha = 1; miniBatch=60;

fprintf('\n\n lr0 = %1.2e, ALPHA = %1.2e, minibatch = %d \n', lr0, alpha, miniBatch);

%% set up optimization
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fTest   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=addBias; fTest.pLoss.addBias=addBias;

vec     = @(x) x(:);

if addBias==false
    W0      = vec(randn(nf,nc));
else
    W0      = vec(randn(nf+1,nc));
end

%% sgd setup
opt           = sgd('out',1);
opt.nesterov=false;
opt.rtol      = 1e-3;
opt.atol      = 1e-3;
opt.miniBatch = miniBatch;
opt.maxEpochs = 100;
opt.learningRate = @(epoch) lr0/sqrt(epoch);
opt.stoppingTime = 500; %seconds


%% solve
tSolve = tic
[Wopt, hisOpt] = solve(opt,f,W0, fTest);
tSolve = toc(tSolve)

fprintf('\n\n ALPHA = %1.2e, fTrain = %1.2e, fTest = %1.2e, trainAcc = %1.2f, testAcc=%1.2f\n\n', ...
    alpha, hisOpt.his(end,9), hisOpt.his(end,13), hisOpt.his(end,10), hisOpt.his(end,13)); %14 for newton

atol = opt.atol; rtol = opt.rtol; 
miniBatch = opt.miniBatch;
save('sgdResultsCIFAR10.mat', 'hisOpt', 'Wopt', 'alpha', 'atol', 'miniBatch', 'rtol', 'lr0', 'miniBatch')