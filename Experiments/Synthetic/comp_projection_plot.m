function fig = comp_projection_plot(algo,N,k,eps,degree,varargin)
%% Description
% This function produces the plots for the comparison between the RMSE of
% chosen algorithm and unconstrained polynomial regression at a certain
% degree for varying noise level eps. It also returns the array
% of RMSE for training and testing for both algorithms
% Outputs:
%   fig := matlab figure with 4 graphs
%   rmse_train_algo = array of rmse for the choosen algorithm on the training
%                    dataset when fitting polynomials of different degrees
% Inputs:
%   algo: the regression algoritm - monotone or convex
%   N: number of examples
%   k: number of features
%   degree: degree of the polynomial regressed
%   varargin: optional argument; a sequence of the form:(or any subset of it)
%                      'monotone_profile', ones(3,1),
%                      'convex_sign', -1,
%                      'l_bound', [-1, 2],
%                      'u_bound', [3, 5]



 
%% Generate synthetic data
switch algo
    case 'monotone'
        real_fun = @monotone_fun;
        algo_abr = 'MPR';
    case 'bounded_derivative'
        real_fun = @bounded_derivative_fun;
        algo_abr = 'BDPR';
    case 'convex'
        real_fun = @convex_fun;
        algo_abr = 'CPR';
    case 'monotone_convex'
        real_fun = @monotone_convex_fun;
        algo_abr = 'MCPR';
    otherwise
        msg="Error: "+algo+" not found, choose between 'monotone' or 'convex'";
        error(msg);
end

% Create the features (generate k x N numbers)
% and scale each feature between 0.5 and 2
features = rand(N,k) * (2-0.5) + 0.5;
% features = normrnd(0,1, [N, k])/3 + 2;

% Generate response variable (clean, no noise added)
response_clean = real_fun(features);

noise = eps*std(response_clean).*normrnd(0,1,[N,1]);
response = response_clean + noise;

%% Train the respective shape-constrained and unconstrained predictor
[p,x] = shape_constrained_regression(algo,degree,features,response,varargin{:});
uncs_mdl = unconstrained_regression(degree,features,response);

%% Create the projected data
proj_features = ones(N,k).*mean(features);
proj_features(:,1) = sort(features(:,1));
proj_response = real_fun(proj_features);

%% Compute predictions for the predicted data
% shape-constrained algorithm
algo_response = zeros(N,1);
for i = 1:N
    algo_response(i) = replace(p, x, proj_features(i,:));
end
% unconstrained algorithm
uncs_response = uncs_mdl.predict(proj_features);
%% Plot the comparative figure and save it to the Plots folder
fig = figure();
c_map = lines(2);
hold on

% plot average values for RMSE
p1 = plot(proj_features(:,1), proj_response,':', 'LineWidth', 2, 'Color','black');
p2 = plot(proj_features(:,1), uncs_response,'-', 'LineWidth', 2, 'Color',c_map(1,:));
p3 = plot(proj_features(:,1), algo_response,'-', 'LineWidth', 2, 'Color',c_map(2,:));

lgd = legend([p1 p2 p3], 'Original function','UPR projection', ...
             strcat(algo_abr, ' projection'));
lgd.Title.String = sprintf('degree = %d', degree);
set(lgd, 'Location', 'Best')
xlabel('Feature_0');
ylabel('Response') 
title(['Comparison of projections for d = ', num2str(degree)]); 
grid on
hold off

% save the figure and a PNG plot 
name = sprintf("proj_%s_N%d_k%d_d%d_eps%.2f",algo_abr,degree,eps);
fig_name = name + '.fig';
png_name = name + '.png';
wd = pwd;
plot_dir = fullfile(wd, 'Plots');
cd(plot_dir);
savefig(fig_name);
saveas(fig,png_name,'png');
cd(wd)

end