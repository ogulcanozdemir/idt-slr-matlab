function [models] = generateGMMs(randomSamples, opts, params, cmpName, cmpDim, splitId)

models = cell(params.num_repeat, 1);
for repeatIdx = 1:params.num_repeat
    subsampledData = cell2mat(randomSamples(repeatIdx));
    subsampledData = subsampledData(:, cmpDim);
    
    fprintf(['Started PCA (repeat:' num2str(repeatIdx) ') : ']); tic;
    [V, ~, M] = pca2(subsampledData, 0.99);
    subsampledData = (subsampledData - repmat(M, size(subsampledData, 1), 1)) * V; toc;
    
    models{repeatIdx}.pca_V = V;
    models{repeatIdx}.pca_M = M;
    
    fprintf(['Started GMM (repeat:' num2str(repeatIdx) ') : ' ]); tic;
    [means, covariances, priors, ll, posteriors] = vl_gmm(subsampledData', params.num_gmm_clusters, 'MaxNumIterations', 10000);
    toc;
    
    models{repeatIdx}.gmm_means = means;
    models{repeatIdx}.gmm_covariances = covariances;
    models{repeatIdx}.gmm_priors = priors;
    models{repeatIdx}.gmm_ll = ll;
    models{repeatIdx}.gmm_posteriors = posteriors;
end

save([opts.experiment.modelPath filesep 'model_k' num2str(params.num_gmm_clusters) ...
    '_r' num2str(params.num_repeat) '_' cmpName '_' params.sampling_method], 'models', '-v7.3');

end