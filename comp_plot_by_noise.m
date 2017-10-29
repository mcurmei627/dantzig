function [mse_train_algo, mse_train_uncs, mse_test_algo, mse_test_uncs, fig]...
    = comp_plot_by_noise(algo,N,k,degree,eps_arr)
%% Description
% This function produces the plots for the comparison between the MSE of
% chosen algorithm and unconstrained polynomial regression at a certain
% degree for varying noise level eps. It also returns the array
% of MSE for training and testing for both algorithms
% Outputs:
%   fig := matlab figure with 4 graphs
%   mse_train_algo = array of mse for the choosen algorithm on the training
%                    dataset when fitting polynomials of different degrees
% Inputs:
%   algo: the regression algoritm - monotone or convex
%   N: number of examples
%   k: number of features
%   degree: degree of the polynomial regressed
%   eps_arr: array of  noise scaling factors
%% Generate artificial data
switch algo
    case 'monotone'
        real_fun = @ladder_fun;
        algo_abr = 'MCPR';
    case 'convex'
        real_fun = @expo_fun;
        algo_abr = 'CPR';
    otherwise
        msg="Error: "+algo+" not found, choose between 'monotone' or 'convex'";
        error(msg);
end

% Create the features (generate k x N numbers)
% and scale each feature between 0.5 and 2
features = rand(N,k) * (2-0.5) + 0.5;

% Generate response variable (clean, no noise added)
response_clean = real_fun(features);
%% Split the  data set into training and testing
% Split into test and train (note that we do not have a validation set)
[idx_train, ~, idx_test]  = dividerand(N, 0.8, 0, 0.2);

features_train = features(idx_train,:);
features_test = features(idx_test,:);

%% Compute the training and testing MSE for chosen algorithm and unconstrained one
N_eps=length(eps_arr);
mse_train_algo = zeros(N_eps,1);
mse_train_uncs = zeros(N_eps,1);
mse_test_algo = zeros(N_eps,1);
mse_test_uncs = zeros(N_eps,1);
for i = 1:N_eps
    % add noise to the response variable
    err = eps_arr(i)*std(response_clean)*(rand(N,1)-0.5);
    response = response_clean + err; 
    response_train = response(idx_train);
    response_test = response(idx_test);
    
    [mse_train_algo(i),mse_test_algo(i)] = ...
        score(algo,degree,features_train,response_train,features_test,response_test);
    [mse_train_uncs(i),mse_test_uncs(i)] = ...
        score('unconstrained',degree,features_train,response_train,features_test,response_test);
end   

%% Plot the comparative figure and save it to the Plots folder
fig = figure();
hold on
plot(eps_arr, mse_train_algo,':', 'LineWidth', 2)
plot(eps_arr, mse_test_algo,'-', 'LineWidth', 2)
plot(eps_arr, mse_train_uncs,':', 'LineWidth', 2)
plot(eps_arr, mse_test_uncs,'-', 'LineWidth', 2)
lgd = legend(strcat(algo_abr, ' train'), strcat(algo_abr, ' test'),'UPR train','UPR test');
lgd.Title.String = sprintf('degree = %d', degree);
set(lgd, 'Location', 'Best')
xlabel(['Value of ' char(949)]); % char(949) is epsilon
ylabel('MSE') 
title(['Comparison of MSE for degree = ', num2str(degree)]); 
grid on
hold off

% save the figure and a PNG plot 
name = sprintf("%svsUPR_N%dk%d%deps%.2f-%.2f",algo_abr,N,k,degree,eps_arr(1), eps_arr(N_eps));
fig_name = name + '.fig';
png_name = name + '.png';
wd = pwd;
plot_dir = fullfile(wd, 'Plots');
cd(plot_dir);
savefig(fig_name);
saveas(fig,png_name,'png');
cd(wd)

end