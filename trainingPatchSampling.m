clear all; close all; clc;
imdir = './ourBSD/train';
imagefiles = dir([imdir, '/*.jpg']);

npatches = 50000; %number of patches required
insize = 33; outsize = 21; % input-output patch sizes
inhalf = floor(insize/2);outhalf = floor(outsize/2);
inoutdiff = insize - outsize;
wavelet = 'db9'; % wavelet choice
i = 1;
in = zeros(npatches, insize*insize);
out = zeros(npatches, outsize*outsize*3);
while i <= npatches
    % Randomly pick up an image
    fprintf('Generating patch # %d \n', i)
    r = randi([1 size(imagefiles,1)],1,1);
    I = rgb2gray(imread([imdir,'/',imagefiles(r).name]));
    [approx, hor, ver, dia] = dwt2(I,wavelet);
    % Generate bounds for patch sampling
    xbounds = [insize+1, size(approx,1)-insize];
    ybounds = [insize+1,size(approx,2)-insize];
    % Generate patch sampling indices
    xind = randi([xbounds(1) xbounds(2)],1,1);
    yind = randi([ybounds(1) ybounds(2)],1,1);
    % Sample input patch
    inpatch = approx(xind-inhalf:xind+inhalf, yind-inhalf:yind+inhalf);
    %[Gmag,Gdir] = imgradient(patch);
    % Sample output patches
    hpatch = hor(xind-outhalf:xind+outhalf, yind-outhalf:yind+outhalf);
    %hpatch = hpatch(1+inoutdiff/2:insize-inoutdiff/2,...
    %    1+inoutdiff/2:insize-inoutdiff/2);
    vpatch = ver(xind-outhalf:xind+outhalf, yind-outhalf:yind+outhalf);
    %vpatch = vpatch(1+inoutdiff/2:insize-inoutdiff/2,...
    %    1+inoutdiff/2:insize-inoutdiff/2);
    dpatch = dia(xind-outhalf:xind+outhalf, yind-outhalf:yind+outhalf);
    %dpatch = dpatch(1+inoutdiff/2:insize-inoutdiff/2,...
    %    1+inoutdiff/2:insize-inoutdiff/2);
    outpatch = cat(3,hpatch,vpatch,dpatch);
    
    svname = ['./trainX/patch_',num2str(xind),'_',num2str(yind),...
        '_',num2str(i),'.jpg'];
    osvname = ['./trainY/patch_',num2str(xind),'_',num2str(yind),...
        '_',num2str(i),'.mat'];
    imwrite(inpatch,svname);
    save(osvname,'outpatch');
    
    in(i,:) = inpatch(:)';
    out(i,:) = outpatch(:)';
    i = i + 1;
end