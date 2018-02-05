function [handSamples, handSamplesIdx] = generateHandSamples(opts, split, params)

dataPath = opts.dataPath;
featureDim = params.feature_dims;

handSamples = [];
handSamplesIdx = [];
for sampleIdx = 1:size(split, 2)
    fprintf('Sample Idx #%d ...', sampleIdx); tic;
    sample = split(1, sampleIdx);
    
    featurePath = [dataPath filesep sample.class filesep sample.video filesep 'color.features'];
    skeletonPath = [dataPath filesep sample.class filesep sample.video filesep 'skeleton.mat'];
    localFeatures = dlmread(featurePath);
    skeletonFeatures = load(skeletonPath);
    skeletonFeatures = skeletonFeatures.skeleton;
    [localFeaturesTemp, localFeaturesIdx] = checkHandRadius(localFeatures, skeletonFeatures, params.hand_radius);
    handSamples = [handSamples; localFeaturesTemp];
    handSamplesIdx(sampleIdx).name = sample.class;
    handSamplesIdx(sampleIdx).video = sample.video;
    handSamplesIdx(sampleIdx).selectedTraj = localFeaturesIdx;
    toc;
end

end



