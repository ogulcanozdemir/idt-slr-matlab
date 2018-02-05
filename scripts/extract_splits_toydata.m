clear all, close all, clc

dataPath = 'D:\lab-backup\ms-thesis-exp\data\toy-dataset\ToyDataset';
splitPath = 'D:\lab-backup\ms-thesis-exp\data\toy-dataset\ToyDataset_splits';

classDir = dir(dataPath);
classDir = classDir([classDir.isdir]);
classDir(1:2) = [];

%% get all unique users
users = [];
for classIdx = 1:numel(classDir)
    class = classDir(classIdx).name;
       
    videoDir = dir([dataPath filesep class]);
    videoDir = videoDir([videoDir.isdir]);
    videoDir(1:2) = [];
    for videoIdx = 1:numel(videoDir)
       video = videoDir(videoIdx).name;
       nameSplit = strsplit(video, '_');
       user = nameSplit(1);
       users = [users; user];
    end
end
users = unique(users);

%% get splits for each user (each fold)
for userIdx = 1:numel(users)
   user = users{userIdx}; 
   
   train = struct('class', {}, 'video', {}, 'nTrajectory', {});
   test = struct('class', {}, 'video', {}, 'nTrajectory', {});
   trainCounter = 1;
   testCounter = 1;
   for classIdx = 1:numel(classDir)
      class = classDir(classIdx).name;
      
      videoDir = dir([dataPath filesep class]);
      videoDir = videoDir([videoDir.isdir]);
      videoDir(1:2) = [];
      for videoIdx = 1:numel(videoDir)
         video = videoDir(videoIdx).name;
         fprintf('Processing User#%d - %s, Video %s: ', userIdx, user, video); tic;
      
         features = dlmread([dataPath filesep class filesep video filesep 'color.features']);
         
         if strncmpi(video, user, 3) == 1
            test(testCounter).class = class;
            test(testCounter).video = video;
            test(testCounter).nTrajectory = size(features, 1);
            testCounter = testCounter + 1;
         else
            train(trainCounter).class = class;
            train(trainCounter).video = video;
            train(trainCounter).nTrajectory = size(features, 1);
            trainCounter = trainCounter + 1;
         end
         toc;
      end
   end
   fprintf('Saving split #%d - %s: ', userIdx, user); tic;
   save([splitPath filesep 'split' num2str(userIdx) '_user' user '.mat'], ...
       'train', 'test', '-v7.3');
   toc;
end