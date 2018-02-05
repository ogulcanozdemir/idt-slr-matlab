function [modeledData] = prepareFisherVectors(split, opts, params, models, cmpDim, fileName)

modeledData = [];
for sampleIdx = 1:size(split, 2)
    sample = split(sampleIdx);
    classFolder = sample.class;
    videoFolder = sample.video;
    
    fprintf('Preparing data for Class %s and Video %s : ', classFolder, videoFolder); tic;
    
    features = load([opts.dataPath filesep classFolder filesep videoFolder filesep 'color.features']);
    tmpFeature = features(:, cmpDim);
    
    nTraj = size(tmpFeature, 1);
    for repeatIdx = 1:params.num_repeat,
       model = models(repeatIdx, 1); model = model{1, 1};
       tmpDataPca = ((tmpFeature - repmat(model.pca_M, nTraj, 1)) * model.pca_V);
       fisher_data = vl_fisher(tmpDataPca', model.gmm_means, model.gmm_covariances, model.gmm_priors, 'improved');
       modeledData{1, repeatIdx}(sampleIdx).class = classFolder;
       modeledData{1, repeatIdx}(sampleIdx).video = videoFolder;
       modeledData{1, repeatIdx}(sampleIdx).fv = fisher_data;
    end
    toc;
end

save([opts.experiment.fisherPath filesep fileName], 'modeledData', '-v7.3');


end

