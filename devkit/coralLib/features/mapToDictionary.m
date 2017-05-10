function map = mapToDictionary(I, dictionary)
% function map = mapToDictionary(I, dictionary)
% 
% I is column stacked images, basically formatted to work with pdist2
% mapToDictionary maps INPUT image I to INPUT dictionary using the L2
% distance. pdist2 is part of the Piotr Dollar MATLAB toolbox.
% 
%  CREDITS
%  Written and maintained by Oscar Beijbom, UCSD
%  Copyright notice: license.txt
%  Changelog: changelog.txt

map = zeros(size(I, 1), 1);
% divide in 100 parts
P = round(linspace(0, size(I,1), 100));

for i = 2 : length(P)
    thisChunk = I(P(i-1) + 1 : P(i), :);
    dist = pdist2(thisChunk, dictionary, 'sqeuclidean' );
    [~, map(P(i-1) + 1 : P(i), :)] = min(dist, [], 2);
end

end