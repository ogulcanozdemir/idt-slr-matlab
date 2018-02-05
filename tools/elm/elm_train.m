function model = elm_train(T, X, elm_type, params, options)
% function model = elm_train(T, X, elm_type, params, options)
%
% ---INPUTS---
% T                     : Nx1 label vector
% X                     : NxD training features (rows are instances)
% elm_type              : 'KERNEL' (default), 'RANDOM'
% IF 'KERNEL' elm_type
%   params.Reg      : regularization parameter
%   params.K_type   : Kernel type 'lin_kernel', 'RBF_kernel', ... see function kernel_matrix for detail.
%   params.K_para   : vector storing kernel parameters if any. (e.g. empty if 'lin_kernel', gamma if 'RBF_kernel')
% IF 'RANDOM' elm_type
%   params.L        : number of hidden nodes (hidden space dimensionality)
%   params.Activation : type of activation function 'sig', 'sine', 'hardlim', 'tribas', 'radbas'
% options.verbose       : Boolean whether to display messages (default true)
% options.test_on_train : Boolean whether to perform test on training set (default false)
% options.multilabels   :
%
% ---OUTPUT----
% model.Beta            : NxC output of ELM training (C: number of classes)
% model.type
% model.w
% model.b
% model.X
% model.params
%
% ---EXAMPLE USAGE---
% elm_type = 'KERNEL';
% params.Reg = 1;
% params.K_type = 'RBF_kernel';
% params.K_para = [1];
% OR
% elm_type = 'RANDOM';
% params.L = 100;
% params.Activation = 'sig';
% Beta = elm_train(train.labels, train.features, elm_type, params);
% options.verbose = false;
% options.test_on_train = false;
% Beta = elm_train(train.labels, train.features, elm_type, params, options);

if nargin < 5   
    options.test_on_train = false;
    options.verbose       = true;
    options.multilabels   = false;
end
if nargin < 4   
    elm_type      = 'KERNEL';
    params.Reg           = 1;
    params.K_type        = 'lin_kernel';
    params.K_para        = [];
end

[N, D] =size(X);

model.type = elm_type;

if(~options.multilabels)
    label = unique(T);
    C = length(label);
    % Convert label vector to a -1/1 label matrix of size NxC
    [~,T] = ismember(T, label);
    temp_T = zeros(N, C);
    for i = 1:N
        temp_T(i, T(i)) = 1;
    end;
    T=temp_T*2-1;
else
    C = size(T, 2);
end

% Training Phase
tic;
switch elm_type
    case 'KERNEL'
        % Compute kernel matrix of size NxN (caution for large number of instances!)
        if(options.verbose); disp('Computing training kernel'); end
        Omega_train = kernel_matrix(X,params.K_type, params.K_para);
        if(options.verbose); disp('Computing output weights'); end
        model.Beta=((Omega_train+speye(N)/params.Reg)\(T));
        if(~options.test_on_train); clear 'Omega_train'; end
        model.X = X;
        clear 'X';
        model.params.K_type = params.K_type;
        model.params.K_para = params.K_para;
        
    case 'RANDOM'
        % Compute random weights and biases (w, b)
        model.w = rand(params.L,D)*2-1; % input weight
        model.b = rand(params.L,1);           % bias of hidden neurons
        if(options.verbose); disp('Computing random projection of training'); end
        H = bsxfun(@plus, model.w*X', model.b);
        clear 'X';
        
        % Calculate hidden neuron output matrix H
        if(options.verbose); disp('Computing activation function'); end
        switch lower(params.Activation)
            case {'sig','sigmoid'}
                H = 1 ./ (1 + exp(-H));
            case {'sin','sine'}
                H = sin(tempH);
            case {'hardlim'} % Hard Limit
                H = double(hardlim(H));
            case {'tribas'} % Triangular basis function
                H = tribas(H);
            case {'radbas'} % Radial basis function
                H = radbas(H);
        end
        % Calculate output weights
        if(options.verbose); disp('Computing output weights'); end
        %pinv(H') * T; % implementation without regularization factor //refer to 2006 Neurocomputing paper
        model.Beta = ( eye(params.L)/params.Reg + H*H' ) \ H * T;
        model.params.Activation = params.Activation;
end
TrainingTime=toc;
if(options.verbose); disp(['Training time: ' num2str(TrainingTime) ' seconds.']); end

% Calculate training classification accuracy
if(options.test_on_train)
    % Calculate the training output
    Y=(Omega_train * model.Beta);
    [~, T_vector]=max(T, [], 2);
    [~, Y_vector]=max(Y, [], 2);
    TrainingAccuracy=sum(Y_vector == T_vector)./N;
    if(options.verbose);
        disp(['Training accuracy: ' num2str(TrainingAccuracy*100) '%.']);
    end
end
