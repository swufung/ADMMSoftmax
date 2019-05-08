function[Dtrain,Ctrain,Dval,Cval] = setupCIFAR10AlexNet(nTrain, nVal, layer)
% sets up extracted feature matrix from AlexNet using CIFAR10 dataset. THe
% extracted features are automatically split into 5 batches.

% inputs:
%   nTrain = number of training examples
%   nVal   = number of validation examples
%   layer  = layer in AlexNet (default is pool5)

%   outputs: 
%   Dtrain = training data
%   Ctrain = training labels
%   Dval   = validation data
%   Cval   = validation labels

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end

if not(exist('nVal','var')) || isempty(nVal)
    nVal = ceil(nTrain/5);
end

if not(exist('layer','var')) || isempty(layer)
    layer = 'pool5';
end

dataDir = [fileparts(which('Meganet.m')) filesep 'data' filesep 'CIFAR'];

dir1 = horzcat(dataDir, '/transferTrainData1_', int2str(nTrain), '_', layer, '.mat');
dir2 = horzcat(dataDir, '/transferTrainData2_', int2str(nTrain), '_', layer, '.mat');
dir3 = horzcat(dataDir, '/transferTrainData3_', int2str(nTrain), '_', layer, '.mat');
dir4 = horzcat(dataDir, '/transferTrainData4_', int2str(nTrain), '_', layer, '.mat');
dir5 = horzcat(dataDir, '/transferTrainData5_', int2str(nTrain), '_', layer, '.mat');
if not(exist(dir1,'file')) || ...
        not(exist(dir2,'file')) || ...
        not(exist(dir3,'file')) || ...
        not(exist(dir4,'file')) || ...
        not(exist(dir5,'file'))
    warning on
    warning('AlexNet data cannot be found in MATLAB path.')
    
    [Dtrain,Ctrain,Dval,Cval] = setupCIFAR10(nTrain, nVal);
   
    %% propagate features through AlexNet
    % load the network trained with imageNet
    net = alexnet;
    inputSize = net.Layers(1).InputSize;  %227*227*3
                                    

    augimdsTrain = augmentedImageDatastore(inputSize,Dtrain);
    augimdsVal = augmentedImageDatastore(inputSize, Dval);

    fprintf('\n propagating... \n')
    tic()
    featuresTrain = activations(net,augimdsTrain,layer);
    Dval = activations(net,augimdsVal,layer);
    toc()
    
    n1 = 0; 
    n2 = round(nTrain/5);
    n3 = round(2*nTrain/5);
    n4 = round(3*nTrain/5);
    n5 = round(4*nTrain/5);
    n6 = round(5*nTrain/5);
    
    dataDir = [fileparts(which('Meganet.m')) filesep 'data' filesep 'CIFAR'];
    
    % first batch
    data    = featuresTrain(:,:,:,(n1+1):n2);
    labels  = Ctrain(:,(n1+1):n2);
    saveStr = horzcat(dataDir, '/transferTrainData1_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % second batch
    data = featuresTrain(:,:,:,(n2+1):n3);
    labels  = Ctrain(:,(n2+1):n3);
    saveStr = horzcat(dataDir, '/transferTrainData2_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % third batch
    data = featuresTrain(:,:,:,(n3+1):n4);
    labels  = Ctrain(:,(n3+1):n4);
    saveStr = horzcat(dataDir, '/transferTrainData3_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % fourth batch
    data = featuresTrain(:,:,:,(n4+1):n5);
    labels  = Ctrain(:,(n4+1):n5);
    saveStr = horzcat(dataDir, '/transferTrainData4_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % fifth batch
    data = featuresTrain(:,:,:,(n5+1):n6);
    labels  = Ctrain(:,(n5+1):n6);
    saveStr = horzcat(dataDir, '/transferTrainData5_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % testing batch
    data = Dval;
    labels  = Cval;
    saveStr = horzcat(dataDir, '/transferValData_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
end

% Reading in the data

loadStr = horzcat(dataDir, '/transferTrainData1_', int2str(nTrain), '_', layer, '.mat');
load(loadStr)
data1   = double(data);
data1   = reshape(data, [], nTrain/5);
labels1 = labels;

loadStr = horzcat(dataDir, '/transferTrainData2_', int2str(nTrain), '_', layer, '.mat');
data2   = double(data);
data2   = reshape(data, [], nTrain/5);
labels2 = labels;

loadStr = horzcat(dataDir, '/transferTrainData3_', int2str(nTrain), '_', layer, '.mat');
data3   = double(data);
data3   = reshape(data, [], nTrain/5);
labels3 = labels;

loadStr = horzcat(dataDir, '/transferTrainData4_', int2str(nTrain), '_', layer, '.mat');
data4   = double(data);
data4   = reshape(data, [], nTrain/5);
labels4 = labels;

loadStr = horzcat(dataDir, '/transferTrainData5_', int2str(nTrain), '_', layer, '.mat');
data5   = double(data);
data5   = reshape(data, [], nTrain/5);
labels5 = labels;

% testing batch
loadStr = horzcat(dataDir, '/transferValData_', int2str(nTrain), '_', layer, '.mat');
load(loadStr);
Dval   = double(data);
Dval   = reshape(Dval, [], nTrain/5);
Cval   = labels;


Dtrain   = [data1 data2 data3 data4 data5];
Ctrain   = [labels1 labels2 labels3 labels4 labels5];
Dval     = reshape(Dval, [], nVal);
end