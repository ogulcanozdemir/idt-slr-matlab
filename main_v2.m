%% Clear everything
clear all, close all, clc

%% Initialize parameters
opts = setup_opts;
params = setup_hyperparameters;

%% load split
train_split = load('train_split_idt.mat');
test_split = load('test_split_idt.mat');

%% generate random samples for the fold
samples = [];
sampledTrajectories = [];
if strcmp(params.sampling_method, 'random') == 1
   fprintf('Generating random samples for training_split ...'); tic;
   samples = cell(params.num_repeat, 1);
   sampledTrajectories = cell(params.num_repeat, 1);
   for repeatIdx = 1:params.num_repeat
      [samples{repeatIdx}, sampledTrajectories{repeatIdx}] = generateRandomSamples(opts, train_split.trainSplit, params);
   end
   toc;

   fprintf('Saving random samples ...'); tic;
   save([opts.experiment.randomPath filesep 'random_samples_k' num2str(params.num_gmm_clusters) ...
       '_r' num2str(params.num_repeat)], 'samples', 'sampledTrajectories', '-v7.3'); 
   toc;
else
   fprintf('Generating hand samples ...'); tic;
   samples = cell(params.num_repeat, 1);
   sampledTrajectories = cell(params.num_repeat, 1);
   for repeatIdx = 1:params.num_repeat
      [samples{repeatIdx}, sampledTrajectories{repeatIdx}] = generateHandSamples(opts, train_split.trainSplit, params);
   end
   toc;

   fprintf('Saving hand samples ...'); tic;
   save([opts.experiment.handPath filesep 'hand_samples_k' num2str(params.num_gmm_clusters) ...
       '_r' num2str(params.num_repeat)], 'samples', 'sampledTrajectories', '-v7.3'); 
   toc;
end

trainName = ['train_fisher_k' num2str(params.num_gmm_clusters) '_r' num2str(params.num_repeat) '_' params.sampling_method];
testName = ['test_fisher_k' num2str(params.num_gmm_clusters) '_r' num2str(params.num_repeat) '_' params.sampling_method];

%% prepare fisher vectors for the fold
for cmpIdx = 1:numel(params.cmpKeys)
  cmpName = params.cmpKeys{cmpIdx};
  cmpDim = params.cmpVals{cmpIdx};

  fprintf('Preparing data for %s : ', cmpName); tic;
  trainNameCmp = [trainName '_' cmpName];
  testNameCmp = [testName '_' cmpName];

  %% generate gmm models from randomly sampled data
  [models] = generateGMMs(samples, opts, params, cmpName, cmpDim, 1);

  %% prepare train and test fisher vectors for each sign from original samples
  prepareFisherVectors(train_split.trainSplit, opts, params, models, cmpDim, trainNameCmp);
  prepareFisherVectors(test_split.testSplit, opts, params, models, cmpDim, testNameCmp);
  clear modeledData models;

  prepareDataForClassification([opts.experiment.fisherPath filesep trainNameCmp], opts, params, trainNameCmp);
  prepareDataForClassification([opts.experiment.fisherPath filesep testNameCmp], opts, params, testNameCmp);
  toc;
end

prepareReleaseData([opts.experiment.fisherPath filesep trainName], opts, params, trainName);
prepareReleaseData([opts.experiment.fisherPath filesep testName], opts, params, testName);

toc;   
clear randomSamples;
