function [prediction_labels, confidence] = elm_predict(test_data, elm, options)
% ---INPUTS---
% test_data                 : MxD test features (rows are instances)
% elm.Beta                  : NxC output of ELM training (C: number of classes)
% elm.type                  : 'KERNEL', 'RANDOM'
% IF 'KERNEL' params.elm_type
%   elm.X                   : NxD training features (rows are instances)
%   elm.params.K_type       : Kernel type 'lin_kernel', 'RBF_kernel', ... see function kernel_matrix for detail.
%   elm.params.K_para       : vector storing kernel parameters if any. (e.g. empty if 'lin_kernel', gamma if 'RBF_kernel')
% IF 'RANDOM' params.elm_type
%   elm.w                   :
%   elm.b                   :
%   elm.params.Activation   : type of activation function 'sig', 'sine', 'hardlim', 'tribas', 'radbas'
% options.verbose           : Boolean whether to display messages (default true)
%
% ---OUTPUTS----
% prediction_labels         :
% confidence                :
%
% ---EXAMPLE USAGE---
% elm.type = 'KERNEL';
% elm.X    = train_data;
% elm.params.Reg = 1;
% elm.params.K_type = 'lin_kernel';
% elm.params.K_para = [];
% OR
% elm.type = 'RANDOM';
% elm.params.L = 100;
% elm.params.Activation = 'sig';
% elm.Beta = elm_train(train.labels, train.features, elm.type, elm.params, options);
% [prediction_labels, confidence] = elm_predict(test.features, elm);

if nargin < 3
    options.verbose       = true;
end

M = size(test_data, 1);
C = size(elm.Beta, 2);

% Calculate the output of testing input
tic;
switch elm.type
    case 'KERNEL'
        if(options.verbose); disp('Computing test kernel'); end
        H = kernel_matrix(elm.X, elm.params.K_type, elm.params.K_para, test_data);
    case 'RANDOM'
        if(options.verbose); disp('Computing random projection of test'); end
        H = bsxfun(@plus, elm.w*test_data', elm.b);
        clear 'test_data'
        switch lower(elm.params.Activation)
            case {'sig','sigmoid'}
                H = 1 ./ (1 + exp(-H));
            case {'sin','sine'}
                H = sin(tempH_test);
            case {'hardlim'} % Hard Limit
                H = hardlim(H);
            case {'tribas'} % Triangular basis function
                H = tribas(H);
            case {'radbas'} % Radial basis function
                H = radbas(H);
        end
end
Y=(H' * elm.Beta)';
TestingTime=toc;
if(options.verbose)
    disp(['Testing time: ' num2str(TestingTime) ' seconds.']);
end

% Determine labels and calculate confidence
prediction_labels = zeros(M, 1);
confidence = zeros(C, M);
for i = 1:M
    [~, prediction_labels(i)]=max(Y(:,i));
    temp =  logsig(Y(:, i));
    confidence(:, i) = temp/sum(temp);
end
