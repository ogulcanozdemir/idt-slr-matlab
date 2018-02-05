clear all, close all, clc

%% prepare release data
opts = setup_opts;
params = setup_hyperparameters;

%% Prepare fisher vectors for each fold
for loocvIdx = 1:numel(params.loocvKeys)
   loocvId = params.loocvKeys{loocvIdx};
   splitId = params.loocvVals{loocvIdx};
   
   %% load split
   split = load([opts.splitPath filesep 'split' splitId '.mat']);
   
   fprintf('Preparing data for %s-%s : ', loocvId, splitId); tic;
   trainName = ['train_fisher_k' num2str(params.num_gmm_clusters) '_r' num2str(params.num_repeat) '_f' splitId '_'];
   testName = ['test_fisher_k' num2str(params.num_gmm_clusters) '_r' num2str(params.num_repeat) '_f' splitId '_'];
   
   %% prepare data for each component combination
   for cmpIdx = 2:numel(params.cmpKeys)
       cmpComb = combntns(params.cmpKeys, cmpIdx);
       
       for combIdx = 1:size(cmpComb, 1)
          comb = cmpComb(combIdx, :);
          

          siz = 0;
          for idx = 1:numel(comb)
             features = load([opts.experiment.fisherPath filesep trainName comb{idx}]); 
             features = features.modeledData{1, 1};
             data = [data, cell2mat({features.fv})'];   
             if idx == 1
                 for sampleIdx = 1:size(features, 2),
                    labels = [labels; params.classMap(features(sampleIdx).class)];
                end 
             end
          end

          clear data label;
          save([opts.experiment.releasePath filesep  '_' num2str(repeatIdx)], 'data', 'labels', '-v7.3');
       end   
   end
   
end