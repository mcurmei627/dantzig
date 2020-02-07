function [rmse_train_algo, rmse_train_uncs, rmse_test_algo, rmse_test_uncs, fig]...
    = comp_plot_by_degree(algo,N,k,d_min,d_max,eps, N_trials, varargin)
%% Description
% This function produces the plots for the comparison between the RMSE of
% chosen algorithm and unconstrained polynomial regression for varying
% polynomial degrees at the noise level eps. It also returns the array
% of RMSE for training and testing for both algorithms
% Outputs:
%   fig := matlab figure with 4 graphs
%   rmse_train_algo = array of rmse for the choosen algorithm on the training
%                    dataset when fitting polynomials of different degrees
% Inputs:
%   algo: the regression algoritm - monotone or convex
%   N: number of examples
%   k: number of features
%   d_min: min degree of the polynomial regressed
%   d_max: max degree
%   eps: noise scaling factor

%% Initialize return variables
N_degrees=d_max-d_min+1;
rmse_train_algo = zeros(N_trials, N_degrees);
rmse_train_uncs = zeros(N_trials, N_degrees);
rmse_test_algo = zeros(N_trials, N_degrees);
rmse_test_uncs = zeros(N_trials, N_degrees);

%% Run each trial
for trial = 1:N_trials  
    %% Generate artificial data
    switch algo
        case 'monotone'
            real_fun = @convex_fun;
            algo_abr = 'MPR';
        case 'bounded_derivative'
            real_fun = @convex_fun;
            algo_abr = 'BDPR';
        case 'convex'
            real_fun = @convex_fun;
            algo_abr = 'CPR';
        case 'monotone_convex'
            real_fun = @convex_fun;
            algo_abr = 'MCPR';
        otherwise
            msg="Error: "+algo+" not found";
            error(msg);
    end

    % Create the features (generate k x N numbers)
    % and scale each feature between 0.5 and 2
    % features = rand(N,k) * (2-0.5) + 0.5;
    features = normrnd(0,1, [N, k])/3 + 2;

    % Generate response variable (true function + error term)
    response = real_fun(features);

    % Add gaussian noise to the features
    noise = eps*std(features).*normrnd(0,1, [N,k]);
    features = features + noise;

    %% Split the  data set into training and testing
    % Split into test and train (note that we do not have a validation set)
    [idx_train, ~, idx_test]  = dividerand(N, 0.8, 0, 0.2);

    features_train = features(idx_train,:);
    features_test = features(idx_test,:);

    response_train = response(idx_train);
    response_test = response(idx_test);

    %% Compute the training and testing RMSE for chosen algorithm and unconstrained one
    for degree = d_min:d_max
        [rmse_train_algo(trial, degree-d_min+1),rmse_test_algo(degree-d_min+1)] = ...
            score(algo,degree,features_train,response_train,...
                  features_test, response_test, varargin{:});
        [rmse_train_uncs(trial, degree-d_min+1),rmse_test_uncs(degree-d_min+1)] = ...
            score('unconstrained',degree,features_train,response_train,...
                  features_test, response_test);
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
fill([d_min:d_max fliplr(d_min:d_max)],[ci_sup fliplr(ci_inf)], ...
     c_map(1,:), 'LineStyle','none','FaceAlpha', 0.3)

% do the same for the algo data
se = std(rmse_test_algo)/sqrt(N_trials);     % Standard Error
ci_inf = mean_test_algo + ts(1)*se;          % Confidence Intervals
ci_sup = mean_test_algo + ts(2)*se; 
fill([d_min:d_max fliplr(d_min:d_max)],[ci_sup fliplr(ci_inf)], ...
     c_map(2,:), 'LineStyle','none','FaceAlpha', 0.3)
 
 
% plot average values for RMSE
p1 = plot(d_min:d_max, mean_train_uncs,':', 'LineWidth', 2, 'Color',c_map(1,:));
p2 = plot(d_min:d_max, mean_test_uncs,'-', 'LineWidth', 2, 'Color',c_map(1,:));
p3 = plot(d_min:d_max, mean_train_algo,':', 'LineWidth', 2, 'Color',c_map(2,:));
p4 = plot(d_min:d_max, mean_test_algo,'-', 'LineWidth', 2, 'Color',c_map(2,:));

lgd = legend([p1 p2 p3 p4], 'UPR train','UPR test', ...
             strcat(algo_abr, ' train'), strcat(algo_abr, ' test'));
 
lgd.Title.String = strcat(char(949), ' = ', num2str(eps));
set(lgd, 'Location', 'Best')
xlabel('Degree of the polynomial'); % char(949) is epsilon
ylabel('RMSE')
set(gca,'xtick', d_min:1:d_max); 
title(['Comparison of RMSE for ', char(949), ' = ', num2str(eps)]); 
grid on
hold off

% save the figure and a PNG plot 
name = sprintf("%svsUPR_N%dk%ddmin%ddmax%deps%.2f",algo_abr,N,k,d_min,d_max,eps);
fig_name = name + '.fig';
png_name = name + '.png';
wd = pwd;
plot_dir = fullfile(wd, 'Plots');
cd(plot_dir);
savefig(fig_name);
saveas(fig,png_name,'png');
cd(wd)
end
