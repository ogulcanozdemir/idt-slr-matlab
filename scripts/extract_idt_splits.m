clear all, close all, clc

dataPath = 'D:\lab-backup\ms-thesis-exp\data\toy-dataset\ToyDataset';
splitPath = 'D:\lab-backup\ms-thesis-exp\data\toy-dataset\ToyDataset_splits';

classDir = dir(dataPath);
classDir = classDir([classDir.isdir]);
classDir(1:2) = [];

train_split = load([splitPath filesep 'train_split_py.mat']);
train_split = train_split.trainSplit;


%% train
for idx = 1:numel(train_split)
    idx
    tr_sample = train_split(idx);
    
    features = dlmread([dataPath filesep tr_sample.class filesep tr_sample.video filesep 'color.features']);
    train_split(idx).nTrajectory = size(features, 1);
end
trainSplit = train_split;
save('train_split_idt.mat', 'trainSplit');


%% test
test_split = load([splitPath filesep 'test_split_py.mat']);
test_split = test_split.testSplit;

for idx = 1:numel(test_split)
    idx
    tr_sample = test_split(idx);
    
    features = dlmread([dataPath filesep tr_sample.class filesep tr_sample.video filesep 'color.features']);
    test_split(idx).nTrajectory = size(features, 1);
end
testSplit = test_split;
save('test_split_idt.mat', 'testSplit');
