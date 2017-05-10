function features = getHistFromTextonMap(textonMap, rowCol, patchSize, nbrTextons)
% function features = getHistFromTextonMap(textonMap, rowCol, patchSize,
% nbrTextons)
%
% getHistFromTextonMap extract histograms from INPUT textonMap at locations
% INPUT rowCol (Nx2 matrix). INPUT patchSize (Mx1 array) determines the
% size of the patches. INPUT nbrTextons is needed in the histogram
% extraction.
%
%
%  CREDITS
%  Written and maintained by Oscar Beijbom, UCSD
%  Copyright notice: license.txt
%  Changelog: changelog.txt

if nargin < 5
    framecrop.do = false;
end

nbrPoints = size(rowCol, 1);
features = [];
[maxRow maxCol] = size(textonMap);
for point = 1 : nbrPoints
    
    for psInd = 1 : length(patchSize)
        
        thisPatchSize = patchSize(psInd);
        
        rows = (rowCol(point, 1) - thisPatchSize) : (rowCol(point, 1) + thisPatchSize);
        cols = (rowCol(point, 2) - thisPatchSize) : (rowCol(point, 2) + thisPatchSize);
        
        rows = rows(rows > 0 & rows <= maxRow);
        cols = cols(cols > 0 & cols <= maxCol);
        
        
        patch = textonMap(rows, cols);
        
        featVector = hist(patch(:), 1 : nbrTextons);
        
        features(point, (psInd - 1) * nbrTextons + 1 : psInd * nbrTextons) = featVector ./ sum(featVector);
        
    end
    
end

end