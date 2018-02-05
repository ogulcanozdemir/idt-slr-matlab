function [ params ] = setup_hyperparameters()

params.num_classes = 10;
params.num_gmm_clusters = 64;
params.num_repeat = 5;

%% Initialize IDT parameters
params.feature_dims = 466; % with trajectory coordinates

%% sampling parameters
params.sampling_method = 'hand';
if strcmp(params.sampling_method, 'random') == 1
    params.sampling_rate = 1000;
else
    params.hand_radius = 30;
    params.num_repeat = 1; % must be (1)
end


%% Prepare component map
% hog - 96 dimensional starts from 71 to 166
% hof - 108 dimensional starts from 167 to 274
% mbh - 192 dimensional starts from 275 to 466
% 'hog_hof', 'hog_mbh', 'hof_mbh' : 71:274, [71:166 275:466], 167:466, 
params.cmpKeys = {'shape', 'hog', 'hof', 'mbh'};
params.cmpVals = {41:70, 71:166, 167:274, 275:466};

%% Prepare Leave-One-Out Cross-Validation parameter map
params.loocvKeys = {'lu1o', 'lu2o', 'lu3o', 'lu4o', 'lu5o', 'lu6o'};
params.loocvVals = {'1_user050', '2_user051', '3_user052', '4_user053', '5_user054', '6_user055'};

%% Prepare class map
classes = {'0001', '0002', '0003', '0005', '0006', '0007', '0008', '0011', '0012', '0014'};
labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
params.classMap = containers.Map(classes, labels);


end

