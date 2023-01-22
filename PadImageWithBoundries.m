function [PaddedImage, padR, padC] = PadImageWithBoundries(I0, N)
% Pad with zeros for a rectangular 2-factored image:
    %padFactor = 2^(N+1);
    padFactor = 2^(8+1);
    [r,c,d] = size(I0);
    padR = 0;
    padC = 0;
    
    if(mod(r,padFactor)~=0)
        padR = padFactor - mod(r,padFactor);
    end
    
    if(mod(c,padFactor)~=0)
        padC = padFactor - mod(c,padFactor);
    end
    
    PaddedImage = padarray(I0,[padR padC 0],'replicate','post');
%     PaddedImage = padarray(I0,[padR padC 0],'replicate','post');
end