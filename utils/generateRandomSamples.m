function [randomSamples, randomlySampledTrajectories] = generateRandomSamples(opts, split, params)
    dataPath = opts.dataPath;
    featureDim = params.feature_dims;
    samplingRate = params.sampling_rate * params.num_gmm_clusters;
    
    randomSamples = zeros(samplingRate, featureDim);
    numTrajectories = sum([split.nTrajectory]);
    
    rng('shuffle', 'v5uniform');
    sampleRange = sort(randsample(numTrajectories, samplingRate));
    
    baseIdx = 0;
    ceilIdx = 0;
    sampleIdx = 1;
    randomlySampledTrajectories = struct('class', {}, 'video', {}, 'trajectories', {});
    for i=1:size(split, 2)
        fprintf('Sampling trajectories from sample id = %d : ', i); tic;
        train_sample = split(i);
        
        featureFilePath = [dataPath filesep train_sample.class filesep train_sample.video filesep 'color.features'];
        features = dlmread(featureFilePath);
        
        randomlySampledTrajectories(i).class = train_sample.class;
        randomlySampledTrajectories(i).video = train_sample.video;
        
        randomlySampledTrajectories(i).trajectories = [];
        ceilIdx = ceilIdx + train_sample.nTrajectory;
        while sampleIdx <= samplingRate && sampleRange(sampleIdx) <= ceilIdx
            localIdx = sampleRange(sampleIdx) - baseIdx;
            randomSamples(sampleIdx,:) = features(localIdx,:);
            randomlySampledTrajectories(i).trajectories = [randomlySampledTrajectories(i).trajectories, localIdx];
            sampleIdx = sampleIdx + 1;
        end
        baseIdx = ceilIdx;
        toc;
    end

end