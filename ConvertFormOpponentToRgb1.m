function [ im ] = ConvertFormOpponentToRgb1( im )

a = 0.2989;
b = 0.587;
c = 0.114;
O1  = im(:,:,1);
O2  = im(:,:,2);
O3  = im(:,:,3);


% B = (sqrt(3)*O3-sqrt(6)*O2)./3;
% G = (sqrt(6)*O2-sqrt(2)*O1+2*B)./2;
% R = sqrt(2)*O1+G;



B = (O3-sqrt(2)/2*(a-b)*O1 - sqrt(6)/2*(a+b).*O2)./(a+b+c);
G = (sqrt(6)*O2-sqrt(2)*O1+2*B)./2;
R = sqrt(2)*O1+G;

% B = (O3-O2)./3;
% G = (O3-O1-B)./2;
% R = O1+G;


im(:,:,1) = R;
im(:,:,2) = G;
im(:,:,3) = B;

end

