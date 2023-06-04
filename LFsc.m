function [out, imNR] = LFsc(im,tat,Fac)
%h = fspecial('motion',Fac , 90-tat*180/pi()); 

%  imagesc(im);figure(gcf);
%cG0 = 2*imfilter(cn,h,'replicate');
%cn_hat=max(cn,cG0);
%im=im./(1+cn_hat);
c50=0.05;%min(0.5,1/(THR+10^-7)*0.15);

imNR=im.^2./(im.^2+c50.^2);%RF
%imNR=im;
h = fspecial('motion',Fac , 90-tat*180/pi()); 

out = 2*imfilter(imNR,h,'replicate');

h = fspecial('motion',round(Fac/2) , 90-tat*180/pi()); 

out =max(Fac/4,1)* (out + imfilter(imNR,h,'replicate'));

%  
% h = fspecial('motion',Fac , 90+45); 
% 
%  imagesc(h);title(tat*180/pi())
%  
%  im=im*.0;
%  im(100:110,100:110)=1;
%  
%  for i=0:25:180
%  h = fspecial('motion',20 , 90-i);
% 
% out = imfilter(im,h,'replicate');
% figure;imagesc(out);title(i)
%  end
% 