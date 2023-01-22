 function [BlurredPyramid, padR, padC]= GenerateBlurredPyramid(I,N)

% Pad with zeros for a rectangular 2-factored image:

[I, padR, padC] = PadImageWithBoundries(I, N);
%[I, padR, padC] = PadImageWithZeros(I, N);


BlurredPyramid = cell(1,N+1);

% Generating Blurred Pyramid

BlurredPyramid{1} = I;
for i=2:N+1
    BlurredPyramid{i} = my_impyramid(BlurredPyramid{i-1} , 'reduce');
end

% BlurredPyramid{1} = NormAndConvertImagetoDouble(I);
% for i=2:N+1
%     BlurredPyramid{i} = my_impyramid(BlurredPyramid{i-1} , 'reduce');
%     BlurredPyramid{i} = NormAndConvertImagetoDouble(BlurredPyramid{i});
% end

% for i=1:N+1
%     BlurredPyramid{i} = NormAndConvertImagetoDouble(BlurredPyramid{i});
% end