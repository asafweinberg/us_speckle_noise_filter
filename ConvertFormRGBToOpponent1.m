function [ im ] = ConvertFormRGBToOpponent1( im )

R  = im(:,:,1);
G  = im(:,:,2);
B  = im(:,:,3);
%convert to opponent space
a = 0.2989;
b = 0.587;
c = 0.114;
O1 = (R-G)./sqrt(2);
O2 = (R+G-2*B)./sqrt(6);
O3 = (a*R+b*G+c*B);


%O1 = -0.0971.*R + 0.1458.*G -0.0250.*B;
%O2 = -0.0930.*R +0.2529.*G + 0.4665.*B
%O3 = 0.2814.*R+ 0.6938.*G +0.0638.*B;

% O1 = (R-G)./1;
% O2 = (R+G-2*B)./1;
% O3 = (R+G+B)./1;

im(:,:,1) = O1;
im(:,:,2) = O2;
im(:,:,3) = O3;

end

