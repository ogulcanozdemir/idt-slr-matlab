function [opts] = setup_opts()

%% initialize path variables
disp('Initializing path variables...'); tic;
opts.vlfeatPath = 'tools\vlfeat\toolbox';
opts.dataPath = 'D:\lab-backup\ms-thesis-exp\data\toy-dataset\ToyDataset';
opts.splitPath = 'D:\lab-backup\ms-thesis-exp\data\toy-dataset\ToyDataset_splits';
disp(opts);

addpath('utils');
addpath('tools');

disp('Initializing experiment variables...');
opts.experimentPath = 'experiments';
opts.experiment.fisherPath = [opts.experimentPath filesep 'fisher'];
opts.experiment.modelPath = [opts.experimentPath filesep 'model'];
opts.experiment.randomPath = [opts.experimentPath filesep 'random'];
opts.experiment.handPath = [opts.experimentPath filesep 'hand'];
opts.experiment.releasePath = [opts.experimentPath filesep 'release'];
disp(opts.experiment);
toc;

%% include paths
addpath(opts.vlfeatPath);
addpath(opts.dataPath);
addpath(opts.splitPath);
addpath(opts.experiment.fisherPath);
addpath(opts.experiment.modelPath);
addpath(opts.experiment.randomPath);
addpath(opts.experiment.releasePath);

%% initialize vlfeat
disp('Initializing vlfeat library...'); tic;
run([opts.vlfeatPath filesep 'vl_setup']);
toc;

end

