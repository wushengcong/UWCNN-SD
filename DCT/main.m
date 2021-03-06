clc,close all,clear all;

inputImgPath = '../Inputs';                 % input image path
if ~exist('HF','dir')
    mkdir('HF')
end
if ~exist('LF','dir')
    mkdir('LF')
end
imgFiles = imdir(inputImgPath);
for indImg = 1:length(imgFiles)    
    % read image
    imgPath = fullfile(inputImgPath, imgFiles(indImg).name);
    img.RGB = im2double(imread(imgPath));
    img.name = imgPath((strfind(imgPath,'Inputs')+7):end);
    [s1,s2,s3] = size(img.RGB);
    High = zeros(s1,s2,s3);
    Low = zeros(s1,s2,s3);
    for i = 1:3       
        Orig = img.RGB(:,:,i);       
        % Transform
        Orig_T = dct2(Orig);
        % Split between high- and low-frequency in the spectrum (*)
        cutoff = round(0.65 * 256);
        High_T = fliplr(tril(fliplr(Orig_T), cutoff));
        Low_T = Orig_T - High_T;
        % Transform back
        High(:,:,i) = idct2(High_T);
        Low(:,:,i) = idct2(Low_T);
    end

    HFPath = fullfile('HF', strcat(img.name(1:end-4), '.png'));  
    imwrite(High,HFPath);
    LFPath = fullfile('LF', strcat(img.name(1:end-4), '.png')); 
    imwrite(Low,LFPath);
end







