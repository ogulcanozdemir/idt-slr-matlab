function prepareDataForClassification(fisherDataFile, opts, params, outputFile)

load(fisherDataFile);
for repeatIdx=1:params.num_repeat,
    tmpData = modeledData{1, repeatIdx};
    
    data = [];
    labels = [];
    for sampleIdx = 1:size(tmpData, 2),
       labels = [labels; str2num(tmpData(sampleIdx).class)];
       data = [data; tmpData(sampleIdx).fv'];
    end    
    clear tmpData;
    save([opts.experiment.releasePath filesep outputFile '_' num2str(repeatIdx)], 'data', 'labels', '-v7.3');
end

end

