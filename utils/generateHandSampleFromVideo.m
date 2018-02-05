function [samples] = generateHandSampleFromVideo(opts, params, localFeatures, class, video)
    skeletonPath = [opts.dataPath filesep class filesep video filesep 'skeleton.mat'];
    skeletonFeatures = load(skeletonPath);
    skeletonFeatures = skeletonFeatures.skeleton;
    [samples, ~] = checkHandRadius(localFeatures, skeletonFeatures, params.hand_radius);
end