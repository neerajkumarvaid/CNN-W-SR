clear all; close all; clc;
imdir = './BSDS300/train';
imagefiles = dir([imdir, '/*.jpg']); 
svdir = './ourBSD/train';
mkdir(svdir);

for i = 1:size(imagefiles,1)
    fprintf('Processing image # %d \n',i);
    name = imagefiles(i).name;
    I = imread([imdir,'/',name]);
    if size(I,1) < size(I,2)
        I = I(1:320,1:480,:);
    else
        I = I(1:480,1:320,:);
    end
    imwrite(I,[svdir,'/',num2str(i),'.jpg']);
end
        