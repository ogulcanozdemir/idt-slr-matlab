function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);

if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type, 'histogram_intersection')
    if nargin<4,
        omega = zeros(nb_data, nb_data);
        for i = 1:nb_data
            omega(:, i) = histogram_intersection(Xtrain(i, :), Xtrain);
        end
    else
    end
    
elseif strcmp(kernel_type, 'chi_squared')
    addpath('C:\Dropbox\Toolboxes\histogram_distance\');
    dist_func=@chi_square_statistics_fast;
    tic;
    if nargin<4
        omega=pdist2(Xtrain,Xtrain,dist_func);
    else
        omega=pdist2(Xtrain,Xt,dist_func);
    end
    toc;
%     if nargin<4
%         omega = zeros(nb_data,nb_data);
%         for i=1:nb_data
%             d = bsxfun(@minus, Xtrain, Xtrain(i,:));
%             s = bsxfun(@plus, Xtrain, Xtrain(i,:));
%             omega(:,i) = sum(d.^2 ./ (s/2+eps), 2);
%         end
%     else
%         omega = zeros(nb_data,size(Xt, 1));
%         for i=1:size(Xt,1)
%             d = bsxfun(@minus, Xtrain, Xt(i,:));
%             s = bsxfun(@plus, Xtrain, Xt(i,:));
%             omega(:,i) = sum(d.^2 ./ (s/2+eps), 2);
%         end
%     end
%     omega = 1 - omega;

%     chi_squared = @(x,y) 1 - sum((x - y).^2 ./ (x + y) / 2);
    %omega = pdist2(Xtrain + eps, Xtrain + eps, chi_squared);
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end
end
function d = histogram_intersection(XI, XJ)
m=size(XJ,1); % number of samples of p
d=zeros(m,1); % initialize output array
  
sxi=sum(XI);
for i=1:m
	d(i,1) = 1 - (sum(min(XI, XJ(i,:))) / sxi);
end
end