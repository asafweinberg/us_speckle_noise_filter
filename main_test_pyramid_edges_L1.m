

imagepath= 'non_watercolor_4_ic.png';

I1 = im2double(imresize(imread(imagepath),0.7));


I = cat(1,I1, im2double(imresize(imread('non_a_watercolor_4_darker.png'),[size(I1,1) size(I1,2)])));



minS = min(size(I,1),size(I,2));


N =2;
Iop  = ConvertFormRGBToOpponent1( I);



L= 0.5*[ 0,-1,0 ;-1,4,-1;0,-1,0];





clear R;

Wv1 = 0;
[BlurredPyramidGray,~, ~, ~] = GenerateBlurredPyramid(ConvertFormRGBToOpponent1(repmat(rgb2gray(I),1,1,3)),N);
[BlurredPyramidWhite,~, ~, ~] = GenerateBlurredPyramid(ConvertFormRGBToOpponent1(ones(size(I))),N);
[BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop(:,:,1)),N);
[h,w] = size(BlurredPyramid{1});
W1 = zeros(h,w,3);
for c = 1:3
    [BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop(:,:,c)),N);

    W = abs(imfilter(abs(BlurredPyramid{N+1}).*(1-0),L,'replicate'));
  
     if(max(W(:))>0)
            W = W./max(W(:));
     end;

    for i = N:-1:1

        
        Gn = abs(imfilter(abs(BlurredPyramid{i}),L,'replicate'));

        W = max(my_impyramid(W,'expand') , Gn);
    
        if(max(W(:))>0)
            W = W./max(W(:));
        end;
        

    end

   W1(:,:,c) = W;
 
end
W = max(W1,[],3);


for c = 1:3
     [BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop(:,:,c)),N);
     [Gh ,Gv] = imgrad(BlurredPyramid{1});
     
    
     
     R(:,:,c) = poisson_solver_function(1*(0.5*(W)+1.0).*Gh,1*(0.5*(W)+1.0).*Gv,(BlurredPyramid{1}));
    
     
end
    


Rrgb = ConvertFormOpponentToRgb1( R );
m = max(Rrgb(:))-1;

FinalImage = Rrgb(1:end-padR,1:end-padC,:);


imtool( FinalImage./max(FinalImage(:)));



% [BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop),N);
% Io = BlurredPyramid{1};
% 
% E = (Io(:,:,1)==0).*( Io(:,:,2)==0);
% %imtool( (E).*Rrgb./max(Rrgb(:)));
% RI = I(145,:,1);
% GI = I(145,:,2);
% BI = I(145,:,3);
% 
% RR = FinalImage(145,:,1);
% GR = FinalImage(145,:,2);
% BR = FinalImage(145,:,3);
% x = 1:1:length(I(145,:,1));
% 
% figure;plot(x,RR,x,RI)
% xlabel('Position[pixels]') % x-axis label
% ylabel('Normalized intensity') %

