function out = rowVector(in)
% function out = rowVector(in)
% 
% returns a rowvector regardless of the shape of the input vector.
% Works only for vectors and scalars!
%
%  CREDITS
%  Written and maintained by Oscar Beijbom, UCSD
%  Copyright notice: license.txt
%  Changelog: changelog.txt

if (size(in,1) > 1 && size(in,2) > 1)
    error('matrix inputs not allowed');
end

if size(in, 1) > 1
    out = in';
else
    out = in;
end


end