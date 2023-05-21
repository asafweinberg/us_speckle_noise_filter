dir = "D:\Asaf\University\year4\Project\us_speckle_noise_filter\metrics\breastTumorDeepLabV3";

exampleDir = fullfile(dir,"breastTumorDeepLabV3");
% loadedModel = load(fullfile(dir,"breast_seg_deepLabV3.mat"));

imTest = imread(fullfile(dir,"breastUltrasoundImg.png"));
 imSize = [256 256];
imTest = imresize(imTest,imSize);

segmentedImg = semanticseg(imTest,trainedNet);