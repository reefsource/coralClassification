function dataOut = featuresClassic(I, labelMatrix, featureParams, featurePrep)
% function dataOut = featuresClassic(cs, fileNbr)
%
% featuresClassis is the high level feature extraction file. It created the
% texton map from INPUT image I and then extract histograms at locations 
% specified by INPUT labelMatrix. INPUT struct featureParams control various
% aspects of the feature extraction process, and INPUT featurePrep contains
% the texton (the dictionary).
%
%  CREDITS
%  Written and maintained by Oscar Beijbom, UCSD
%  Copyright notice: license.txt
%  Changelog: changelog.txt

% resize image and labelMatrix
I = (imresize(I, 1/featureParams.resizeFactor));
labelMatrix(:, 1:2) = round(labelMatrix(:,1:2)./featureParams.resizeFactor);

rowCol = labelMatrix(:, 1:2);

% extract textonMap
textonMap = extractTextonMap(featureParams, featurePrep{1}, I);

% pool textonMap to features.
features = getHistFromTextonMap(textonMap, rowCol, featureParams.patchSize, 135);

% store the results.
nfeatures = size(features, 1);
dataOut.features = single(features);
dataOut.labels = labelMatrix(:, 3);
dataOut.rowCol = rowCol;
dataOut.pointNbr = (1 : nfeatures)';

end
