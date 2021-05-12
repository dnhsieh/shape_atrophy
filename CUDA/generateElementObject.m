function elmObj = generateElementObject(varargin)

if length(varargin) == 3
	elmObj = generateElementObject2D(varargin{:});
elseif length(varargin) == 5
	elmObj = generateElementObject3D(varargin{:});
else
	error('Unknown input format.');
end
