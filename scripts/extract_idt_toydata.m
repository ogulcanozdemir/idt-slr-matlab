clear all, close all, clc

dataPath = 'E:\ms-thesis\data\toy-dataset\ToyDataset';
executablePath = 'E:\ms-thesis\implementation\dense-trajectories\tools\DenseTrackStab.exe';

inputFileSuffix = 'color_scaled.mp4';
outputFileSuffix = 'color.features.gz';

classDir = dir(dataPath);
classDir = classDir([classDir.isdir]);
classDir(1:2) = [];

% class
for classIdx = 1:numel(classDir)
    class = classDir(classIdx).name;
    
    % video
    videoDir = dir([dataPath filesep class]);
    videoDir = videoDir([videoDir.isdir]);
    videoDir(1:2) = [];
    parfor videoIdx = 1:numel(videoDir)
       video = videoDir(videoIdx).name;
       
       fprintf('Extracting Class: %s, Sample: %s ... ', class, video); tic;
       filePath = [dataPath filesep class filesep video filesep];
       %% extract idt
%        execStr = [executablePath ' ' filePath inputFileSuffix ...
%            ' | gzip > ' filePath outputFileSuffix];
       
       %% scale videos
%        execStr = ['ffmpeg -loglevel panic -i ' filePath inputFileSuffix ...
%                     ' -vf scale=640:360 ' filePath outputFileSuffix];
%        system(execStr);
       gunzip([filePath outputFileSuffix]);
       toc;
    end
end