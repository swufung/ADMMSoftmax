function[Dtrain,Ctrain,Dval,Cval,Dtest,Ctest] = setupCIFAR10AlexNet(nTrain, nTest, layer)
% sets up extracted feature matrix from AlexNet using CIFAR10 dataset. THe
% extracted features are automatically split into 5 batches.
% 5h batch used for validation

% inputs:
%   nTrain = number of training examples
%   nVal   = number of validation examples
%   nTest  = number of testing examples
%   layer  = layer in AlexNet (default is pool5)

%   outputs: 
%   Dtrain = training data
%   Ctrain = training labels
%   Dval   = validation data
%   Cval   = validation labels

if not(exist('nTrain','var')) || isempty(nTrain)
    nTrain = 50000;
end

if not(exist('nTest','var')) || isempty(nTest)
    nTest = ceil(nTrain/5);
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
    
    [Dtrain,Ctrain,Dtest,Ctest] = setupCIFAR10(nTrain, nTest);
   
    %% propagate features through AlexNet
    % load the network trained with imageNet
    net = alexnet;
    inputSize = net.Layers(1).InputSize;  %227*227*3
                                    

    augimdsTrain = augmentedImageDatastore(inputSize,Dtrain);
    augimdsTest = augmentedImageDatastore(inputSize, Dtest);

    fprintf('\n propagating... \n')
    tic()
    featuresTrain = activations(net,augimdsTrain,layer);
    Dtest = activations(net,augimdsTest,layer);
    toc()
    
    n1 = 0; 
    n2 = round(nTrain/5);
    n3 = round(2*nTrain/5);
    n4 = round(3*nTrain/5);
    n5 = round(4*nTrain/5);
    n6 = round(5*nTrain/5);
    
    featuresTrain1 = featuresTrain(:,:,:,(n1+1):n2);
    featuresTrain2 = featuresTrain(:,:,:,(n2+1):n3);
    featuresTrain3 = featuresTrain(:,:,:,(n3+1):n4);
    featuresTrain4 = featuresTrain(:,:,:,(n4+1):n5);
    featuresTrain5 = featuresTrain(:,:,:,(n5+1):n6);
    
    labels1  = Ctrain(:,(n1+1):n2);
    labels2  = Ctrain(:,(n2+1):n3);
    labels3  = Ctrain(:,(n3+1):n4);
    labels4  = Ctrain(:,(n4+1):n5);
    labels5  = Ctrain(:,(n5+1):n6);
    
    % check that they are different
    norm1 = norm(featuresTrain1(:)-featuresTrain2(:));
    norm2 = norm(featuresTrain1(:)-featuresTrain3(:));
    norm3 = norm(featuresTrain1(:)-featuresTrain4(:));
    norm4 = norm(featuresTrain1(:)-featuresTrain5(:));
    norm5 = norm(featuresTrain2(:)-featuresTrain5(:));
    
    fprintf('differences = %1.2e, %1.2e, %1.2e, %1.2e, %1.2e',...
        norm1, norm2, norm3, norm4, norm5);
    
    
    
    
    
    dataDir = [fileparts(which('Meganet.m')) filesep 'data' filesep 'CIFAR'];
    
    % first batch
%     data    = featuresTrain(:,:,:,(n1+1):n2);
%     labels  = Ctrain(:,(n1+1):n2);
    data = featuresTrain1;
    labels = labels1;
    saveStr = horzcat(dataDir, '/transferTrainData1_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % second batch
%     data = featuresTrain(:,:,:,(n2+1):n3);
%     labels  = Ctrain(:,(n2+1):n3);
    data = featuresTrain2;
    labels = labels2;
    saveStr = horzcat(dataDir, '/transferTrainData2_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % third batch
%     data = featuresTrain(:,:,:,(n3+1):n4);
%     labels  = Ctrain(:,(n3+1):n4);
    data = featuresTrain3;
    labels = labels3;
    saveStr = horzcat(dataDir, '/transferTrainData3_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % fourth batch
%     data = featuresTrain(:,:,:,(n4+1):n5);
%     labels  = Ctrain(:,(n4+1):n5);
    data = featuresTrain4;
    labels = labels4;
    saveStr = horzcat(dataDir, '/transferTrainData4_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % fifth batch
%     data = featuresTrain(:,:,:,(n5+1):n6);
%     labels  = Ctrain(:,(n5+1):n6);
    data = featuresTrain5;
    labels = labels5;
    saveStr = horzcat(dataDir, '/transferTrainData5_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
    
    % testing batch
    data = Dtest;
    labels  = Ctest;
    saveStr = horzcat(dataDir, '/transferTestData_', int2str(nTrain), '_', layer, '.mat');
    save(saveStr, 'data', 'labels');
end

% Reading in the data

loadStr = horzcat(dataDir, '/transferTrainData1_', int2str(nTrain), '_', layer, '.mat');
load(loadStr)
data1   = double(data);
data1   = reshape(data, [], nTrain/5);
labels1 = labels;

loadStr = horzcat(dataDir, '/transferTrainData2_', int2str(nTrain), '_', layer, '.mat');
load(loadStr)
data2   = double(data);
data2   = reshape(data, [], nTrain/5);
labels2 = labels;

loadStr = horzcat(dataDir, '/transferTrainData3_', int2str(nTrain), '_', layer, '.mat');
load(loadStr)
data3   = double(data);
data3   = reshape(data, [], nTrain/5);
labels3 = labels;

loadStr = horzcat(dataDir, '/transferTrainData4_', int2str(nTrain), '_', layer, '.mat');
load(loadStr)
data4   = double(data);
data4   = reshape(data, [], nTrain/5);
labels4 = labels;

loadStr = horzcat(dataDir, '/transferTrainData5_', int2str(nTrain), '_', layer, '.mat');
load(loadStr)
data5   = double(data);
data5   = reshape(data, [], nTrain/5);
labels5 = labels;

% testing batch
loadStr = horzcat(dataDir, '/transferTestData_', int2str(nTrain), '_', layer, '.mat');
load(loadStr);
Dtest   = double(data);
% Dtest   = reshape(Dtest, [], nTrain/5);
Ctest   = labels;


Dtrain   = [data1 data2 data3 data4];
Ctrain   = [labels1 labels2 labels3 labels4];

% use fifth batch for validation
Dval     = data5; Cval = labels5;

Dtest     = reshape(Dtest, [], nTest);
end