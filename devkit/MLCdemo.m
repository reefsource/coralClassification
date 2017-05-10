%% GENERAL INSTRUCTIONS %%
%
% This is the development kit for the MLC dataset. 
% This demo file demonstrate how to extract feature-descriptors 
% and train classifiers as implemented by 
% [Automated Annotation of Coral Reef Survey Images] Beijbom et. al.
%
%  CREDITS
%  Written and maintained by Oscar Beijbom, UCSD
%  Copyright notice: license.txt
%  Changelog: changelog.txt

%% SETUP %%

% (0) Add coralLib with subdirectories to MATLAB path
addpath(genpath('./coralLib'));

% (1) Download data from http://vision.ucsd.edu/content/moorea-labeled-corals
% There are three files, 2008.zip, 2009.zip and 2010.zip. 
% Unzip and MOVE ALL IMAGES INTO ONE FOLDER. Set path:
dataDir = '/Users/hblasinski/Desktop/Moorea/images';

% (2) Download Piotr Dollars toolbox from http://vision.ucsd.edu/~pdollar/toolbox/doc/
addpath(genpath('/Users/hblasinski/Desktop/Moorea/toolbox'));

% (3) Download Libsvm from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
% For speed reasons we use the libsvm compiled binaries to training and
% testing (NOT THE MATLAB INTERFACE). Please enter the path to those here
solverTrainPath = '/Users/hblasinski/Desktop/Moorea/libsvm/svm-train';
solverTestPath = '/Users/hblasinski/Desktop/Moorea/libsvm/svm-predict';
% We use libsvmwrite.mex (in the /MATLAB directory) to write the data, 
% so make sure this is compiled and on the matlab path.
addpath('/Users/hblasinski/Desktop/Moorea/libsvm/matlab');

% (4) load meta-data structure
load meta.mat

% (5) set working directory for experiment
experimentDir = '/Users/hblasinski/Desktop/Moorea/descriptors';

% (6) set experiment to run, 
% 1 : 2008 => 2008, 
% 2 : 2008 => 2009, 
% 3 : 2008 + 2009 => 2010,
% 4 : test experiment, 
% 5 : 2008 + 2009 + 2010
experimentNbr = 5;

% (7) OPTIONAL
% If you are running this code on other image datasets, you need to set the
% pixel-cm ratio, meaning the number of pixels in the image that cover each
% centimeter of actual bottom distance. To do this set
imheight_cm = 65; %This is the number of pixels the image cover from top to bottom
nrows = 2200; %Number of rows per image.
meta.featureParams.resizeFactor = standardSSrate(imheight_cm, nrows);


%% PLOT AN IMAGE %%
fileNbr = 1;
I = imread(fullfile(dataDir, meta.content(fileNbr).imgFile));
labelMatrix = readCoralLabelFile(fullfile(dataDir, meta.content(fileNbr).labelFile), meta.labelParams);
showCoralImg(I, labelMatrix)

%% EXTRACT FEATURES %%

% select experiment
experiment = meta.experiment(experimentNbr);
fileNbrs = [experiment.trainIds experiment.testIds];

startTime = clock;
fprintf(1, 'Running experiment: %s\n', experiment.toString);
for fileItt = 1 : numel(fileNbrs)
    
    fileNbr = fileNbrs(fileItt);
    fprintf(1, 'Processing file: %d... ', fileNbr);
    
    % load image and annotations
    I = imread(fullfile(dataDir, meta.content(fileNbr).imgFile));
    labelMatrix = readCoralLabelFile(fullfile(dataDir, meta.content(fileNbr).labelFile), meta.labelParams);

    % extract features
    data = featuresClassic(I, labelMatrix, meta.featureParams, meta.featurePrep);
    data.fromfile = repmat(fileNbr, numel(data.labels), 1);
    
    % write to disk
    save(fullfile(experimentDir, sprintf('data%d.mat', fileNbr)), 'data');
    
    estimateRemainingTime(startTime, clock, numel(fileNbrs), fileItt, 1);
    
end


%% TRAIN CLASSIFIER %%

% load feature data from disk
trainData = collectFeatures(experimentDir, experiment.trainIds);
testData = collectFeatures(experimentDir, experiment.testIds);

% normalize test and train data (this is recommended by the libsvm authors)
normalizer = max(trainData.features);
trainData.features = trainData.features ./ repmat(normalizer, numel(trainData.labels), 1);
testData.features = testData.features ./ repmat(normalizer, numel(testData.labels), 1);

% subsample train data to a pre-determined number of features per class.
[trainingWeights ssStats] = getSVMssfactor(trainData, meta.svmParams.targetNbrSamplesPerClass);
trainData = subsampleDataStruct(trainData, trainingWeights);
     
% write train and test data to disk.
libsvmwrite(GetFullPath(fullfile(experimentDir, 'train.dat')), trainData.labels, sparse(double(trainData.features)));
libsvmwrite(GetFullPath(fullfile(experimentDir, 'test.dat')), testData.labels, sparse(double(testData.features)));

% set the options string for the classifier.
optStr = makeSolverOptionString(meta.svmParams.options, trainingWeights);

% Train classifier
system(sprintf('%s %s %s %s', solverTrainPath, optStr, fullfile(experimentDir, 'train.dat'), fullfile(experimentDir, 'model.dat')));

% Test on test data
system(sprintf('%s %s %s %s', solverTestPath, fullfile(experimentDir, 'test.dat'), fullfile(experimentDir, 'model.dat'), fullfile(experimentDir, 'label.dat')));

% Evaluate performance
estLabels = dlmread(fullfile(experimentDir, 'label.dat'));

CM = confMatrix(testData.labels, estLabels, 9);
accuracy = sum(diag(CM)) ./ sum(CM(:));
fprintf('Accuracy is %.2f%%\n', 100*accuracy);
figure(2);
confMatrixShow(CM, meta.labelParams.cats);

