function [isCoralMap, regionProposals, regionLabels] = segmentImage( I, varargin )

p = inputParser;
p.addRequired('I');
p.addOptional('nYsamples',20);
p.addOptional('nXsamples',30);
p.addOptional('imHeight',65);
p.addOptional('nRows',2200);
p.addOptional('experimentDir','.');
p.addOptional('path','.');
p.addOptional('solverTestPath','./libsvm/svm-predict');
p.addOptional('nClasses',9);
p.addOptional('minRegionAreaFraction',0.1);
p.addOptional('currentPixelCmRatio',5.26);

p.parse(I,varargin{:});
inputs = p.Results;

load(fullfile(inputs.path,'devkit','meta.mat'));
TARGET_PIXEL_CM_RATIO = 17.2;
% subSampleRate = thisPixelCmRatio / TARGET_PIXEL_CM_RATIO;
meta.featureParams.resizeFactor = 2*inputs.currentPixelCmRatio/TARGET_PIXEL_CM_RATIO;
%meta.featureParams.resizeFactor = standardSSrate(inputs.imHeight, inputs.nRows);
meta.labelParams.nonCoralId = [1 2 3 4];
meta.labelParams.coralId = [5 6 7 8 9];

% Load the image
h = size(I,1);
w = size(I,2);

% Generate sampling grid
deltaY = h/inputs.nYsamples;
deltaX = w/inputs.nXsamples;

ySamples = linspace(deltaY/2,h-deltaY/2,inputs.nYsamples);
xSamples = linspace(deltaX/2,w-deltaX/2,inputs.nXsamples);

[xx, yy] = meshgrid(ySamples,xSamples);
labelMatrix = round([xx(:), yy(:), ones(numel(xx),1)]);

% extract features
data = featuresClassic(I, labelMatrix, meta.featureParams, meta.featurePrep);

showCoralImg(I,labelMatrix)

%% Classify

% normalize test and train data (this is recommended by the libsvm authors)
load(fullfile(inputs.path,'descriptors','normalizer.mat'));
data.features = data.features ./ repmat(normalizer, numel(data.labels), 1);



% Test on test data
libsvmwrite(GetFullPath(fullfile(inputs.experimentDir, 'test.dat')), data.labels, sparse(double(data.features)));
system(sprintf('%s %s %s %s', fullfile(inputs.path,'libsvm','svm-predict'), fullfile(inputs.experimentDir, 'test.dat'), fullfile(inputs.path,'descriptors', 'model.dat'), fullfile(inputs.experimentDir, 'label.dat')));

% Evaluate performance
estLabels = dlmread(fullfile(inputs.experimentDir, 'label.dat'));

estLabelMatrix = labelMatrix;
estLabelMatrix(:,3) = estLabels;

% showCoralImg(I,estLabelMatrix);


%% Generate ROIs

labelImg = reshape(estLabels,inputs.nXsamples,inputs.nYsamples)';
labelImg = imresize(labelImg,[h,w],'nearest');

% Convert classes to coral presence indicator
isCoralMap = zeros(size(labelImg));
for j=1:length(meta.labelParams.coralId)
    isCoralMap = isCoralMap | labelImg == meta.labelParams.coralId(j);
end

figure; imagesc(isCoralMap);

regionProposals = [];
regionLabels = [];
for j=0:1
    map = isCoralMap == j;
    rprops = regionprops(map);
    regionProposals = cat(1,regionProposals,rprops);
    regionLabels = cat(1,regionLabels,j*ones(length(rprops),1));
end

isCoralMap = imresize(isCoralMap,[h w], 'nearest');

end

