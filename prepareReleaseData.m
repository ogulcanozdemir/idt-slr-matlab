function [ output_args ] = prepareReleaseData(fisherDataFile, opts, params, outputFile)

% for cmpIdx = 1:numel(params.cmpKeys)
%     cmp = params.cmpKeys{cmpIdx};
% 
%     features = load([fisherDataFile '_' cmp]);
%     for repeatIdx=1:params.num_repeat,
%          data = [];
%          labels = [];
%          features_tmp = features.modeledData{1, repeatIdx};
%          data = [data, cell2mat({features_tmp.fv})'];   
%          for sampleIdx = 1:size(features_tmp, 2),
%             labels = [labels; str2num(features_tmp(sampleIdx).class)];
%          end
%          save([opts.experiment.releasePath filesep outputFile '_' cmp '_' num2str(repeatIdx)], 'data', 'labels', '-v7.3');         
%     end
% end


features_hog = load([fisherDataFile '_hog']);
features_hof = load([fisherDataFile '_hof']);
features_mbh = load([fisherDataFile '_mbh']);
for repeatIdx = 1:params.num_repeat
   data = [];
   labels = [];
   features_tmp_hog = features_hog.modeledData{1, repeatIdx};
   features_tmp_hof = features_hof.modeledData{1, repeatIdx};
   features_tmp_mbh = features_mbh.modeledData{1, repeatIdx};
   data = [data, cell2mat({features_tmp_hog.fv})', cell2mat({features_tmp_hof.fv})', cell2mat({features_tmp_mbh.fv})'];
   for sampleIdx = 1:size(features_tmp_hog, 2),
      labels = [labels; str2num(features_tmp_hog(sampleIdx).class)];
   end
   save([opts.experiment.releasePath filesep outputFile '_all_' num2str(repeatIdx)], 'data', 'labels', '-v7.3');         
end

end

