function [rmse_train_algo, rmse_train_uncs, rmse_test_algo, rmse_test_uncs, fig]...
    = comp_plot_by_noise(algo,N,k,degree,eps_arr,N_trials,varargin)
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
%   eps_arr: array of  noise scaling factors
%   N_trials: number of trials for each epsilon value, in order to build
%   confidence intervals
%   varargin: optional argument; a sequence of the form:(or any subset of it)
%                      'monotone_profile', ones(3,1),
%                      'convex_sign', -1,
%                      'l_bound', [-1, 2],
%                      'u_bound', [3, 5]


%% Initialize return variables
N_eps=length(eps_arr);
rmse_train_algo = zeros(N_trials, N_eps);
rmse_train_uncs = zeros(N_trials, N_eps);
rmse_test_algo = zeros(N_trials, N_eps);
rmse_test_uncs = zeros(N_trials, N_eps);

%% Run each trial
for trial = 1:N_trials    
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
    %% Split the  data set into training and testing
    % Split into test and train (note that we do not have a validation set)
    [idx_train, ~, idx_test]  = dividerand(N, 0.8, 0, 0.2);
    features_train = features(idx_train,:);
    features_test = features(idx_test,:);

    %% Compute the training and testing RMSE for chosen algorithm and unconstrained one
    for i = 1:N_eps
        % add noise to the response variable
        noise = eps_arr(i)*std(response_clean).*normrnd(0,1,[N,1]);
        response_noise = response_clean + noise;
        response_train = response_noise(idx_train);
        response_test = response_noise(idx_test);
        
        [rmse_train_algo(trial,i),rmse_test_algo(trial,i)] = ...
            score(algo,degree,features_train,response_train,...
                  features_test,response_test,varargin{:});
        [rmse_train_uncs(trial,i),rmse_test_uncs(trial,i)] = ...
            score('unconstrained',degree,features_train,response_train,...
                  features_test,response_test);
    end   
end
%% Plot the comparative figure and save it to the Plots folder
fig = figure();
c_map = lines(2);
hold on
% compute average train and test performances
mean_train_uncs = mean(rmse_train_uncs);
mean_test_uncs = mean(rmse_test_uncs);
mean_train_algo = mean(rmse_train_algo);
mean_test_algo = mean(rmse_test_algo);

% compute the 90% confidence intervals
se = std(rmse_test_uncs)/sqrt(N_trials);     % Standard Error
ts = tinv([0.05  0.95],N_trials-1);          % T-Score
ci_inf = mean_test_uncs + ts(1)*se;          % Confidence Intervals
ci_sup = mean_test_uncs + ts(2)*se; 
% add shaded confidence interval
fill([eps_arr fliplr(eps_arr)],[ci_sup fliplr(ci_inf)], ...
     c_map(1,:), 'LineStyle','none','FaceAlpha', 0.3)

% do the same for the algo data
se = std(rmse_test_algo)/sqrt(N_trials);     % Standard Error
ci_inf = mean_test_algo + ts(1)*se;          % Confidence Intervals
ci_sup = mean_test_algo + ts(2)*se; 
fill([eps_arr fliplr(eps_arr)],[ci_sup fliplr(ci_inf)], ...
     c_map(2,:), 'LineStyle','none','FaceAlpha', 0.3)

% plot average values for RMSE
p1 = plot(eps_arr, mean_train_uncs,':', 'LineWidth', 2, 'Color',c_map(1,:));
p2 = plot(eps_arr, mean_test_uncs,'-', 'LineWidth', 2, 'Color',c_map(1,:));
p3 = plot(eps_arr, mean_train_algo,':', 'LineWidth', 2, 'Color',c_map(2,:));
p4 = plot(eps_arr, mean_test_algo,'-', 'LineWidth', 2, 'Color',c_map(2,:));

lgd = legend([p1 p2 p3 p4], 'UPR train','UPR test', ...
             strcat(algo_abr, ' train'), strcat(algo_abr, ' test'));
lgd.Title.String = sprintf('degree = %d; N = %d', degree, N);
set(lgd, 'Location', 'Best')
xlabel(['Value of ' char(949)]); % char(949) is epsilon
ylabel('RMSE') 
title(['Comparison of RMSE for degree = ', num2str(degree)]); 
grid on
hold off

% save the figure and a PNG plot 
name = sprintf("noise_%s_N%d_k%d_d%d_eps%.2f-%.2f",algo_abr,N,k,degree,...
                eps_arr(1), eps_arr(N_eps));
fig_name = name + '.fig';
png_name = name + '.png';
wd = pwd;
plot_dir = fullfile(wd, 'Plots');
cd(plot_dir);
savefig(fig_name);
saveas(fig,png_name,'png');
cd(wd)

end