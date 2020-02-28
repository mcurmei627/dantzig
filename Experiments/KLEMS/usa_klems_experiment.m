%% Read the data
filename = fullfile('usa_wk_mar_2017.csv');
production_data = csvread(filename,1,0); % offset by one row to skip the headers;
industry_codes = readtable('industry_codes.csv');

% loop through the industries
N_industry = length(industry_codes.Industry);
cd_rmse_train_arr = zeros(N_industry, 1);
cd_rmse_test_arr = zeros(N_industry, 1);
mc_rmse_train_arr = zeros(N_industry, 1);
mc_rmse_test_arr = zeros(N_industry, 1);

%%
for i=1:N_industry
    code = industry_codes.Industry(i);
    description = industry_codes.Description{i,1};
    
    % filter the data relevant to the industry
    data = production_data(ismember(production_data(:,3), code), :);
    
    % select the relevant colums in the data
    % feature columns are "CAP_const", "LAB_const", "INT_const"
    % response variable is "GO_const"
    response = data(:,8);
    features = data(:,[9, 10, 11]);

    [N,k] = size(features);
    %% Split the data set into training and testing
    % Split into test and train
    %[idx_train, ~, idx_test]  = dividerand(N, 0.8, 0, 0.2);
    
    % temporal split
    idx_train = [1:floor(N*0.8), N];
    idx_test = floor(N*0.8)+1:N-1;

    features_train = features(idx_train,:);
    features_test = features(idx_test,:);

    response_train = response(idx_train);
    response_test = response(idx_test);
    
    N_train = length(response_train);
    N_test = length(response_test);
    
    %% Fit Cobb-Douglass production function
    mdl = cobb_douglas_fit(features_train, response_train);
    Y_hat_train = exp(mdl.predict(log(features_train)));
    Y_hat_test = exp(mdl.predict(log(features_test)));
    
    % Average squared deviation in the training set
    cd_rmse_train = sqrt(value(transpose(Y_hat_train-response_train)*...
                      (Y_hat_train-response_train))/N_train);
    % Average squared deviation in the testing set
    cd_rmse_test = sqrt(value(transpose(Y_hat_test-response_test)*...
                     (Y_hat_test-response_test))/N_test);
    
    %% Fit monotone_convex regression
    % rescale for numerical reasons
    scaled_features_train = features_train/100000;
    scaled_features_test = features_test/100000;
    scaled_response_train = response_train/100000;
    scaled_response_test = response_train/100000;
    
    [p,x] = monotone_convex_regression(4,scaled_features_train,...
                                       scaled_response_train,...
                                       ones(k,1), -1); 
    % compute predicted responses for training and testing dataset
    Y_hat_train = zeros(N_train, 1);
    for j = 1:N_train
        Y_hat_train(j) = replace(p, x, scaled_features_train(j,:));
    end
    Y_hat_test = zeros(N_test, 1);
    for j = 1:N_test
        Y_hat_test(j) = replace(p, x, scaled_features_test(j,:));
    end
    Y_hat_train = 100000*Y_hat_train;
    Y_hat_test = 100000*Y_hat_test;
    
    % Average squared deviation in the training set
    mc_rmse_train = sqrt(value(transpose(Y_hat_train-response_train)*...
                      (Y_hat_train-response_train))/N_train);
    % Average squared deviation in the testing set
    mc_rmse_test = sqrt(value(transpose(Y_hat_test-response_test)*...
                     (Y_hat_test-response_test))/N_test);
    
    %% Print error values
    disp(description);
    cd_msg = sprintf("Cobb-Douglas train error: %.2f, test error: %.2f",...
                     cd_rmse_train, cd_rmse_test);
    disp(cd_msg);
    mc_msg = sprintf("Monotone-Concave train error: %.2f, test error: %.2f",...
                     mc_rmse_train, mc_rmse_test);
    disp(mc_msg);
    disp('********************************************');
    
    cd_rmse_train_arr(i) = cd_rmse_train;
    cd_rmse_test_arr(i) = cd_rmse_test;
    mc_rmse_train_arr(i) = mc_rmse_train;
    mc_rmse_test_arr(i) = mc_rmse_test;
    
end

%% Compare the algorithms

%% RMSE comparison
[mc_rmse_test_sorted,sortIdx] = sort(mc_rmse_test_arr);
cd_rmse_test_sorted = cd_rmse_test_arr(sortIdx);

% sort B using the sorting index
labels = industry_codes.Description(sortIdx);
% remove nans
nanIdx = ~isnan(cd_rmse_test_sorted) & ~isnan(mc_rmse_test_sorted);
cd_rmse_test_sorted = cd_rmse_test_sorted(nanIdx);
mc_rmse_test_sorted = mc_rmse_test_sorted(nanIdx);

fig = figure();
c_map = lines(3);
hold on
b1 = bar(mc_rmse_test_sorted,'stack');
b1.FaceColor = c_map(1,:);
b1.EdgeAlpha = 0;
b2 = bar(cd_rmse_test_sorted,'stack');
b2.EdgeColor = c_map(3,:);
b2.FaceColor = c_map(3,:);
b2.FaceAlpha = 0.3;
b2.EdgeAlpha = 1;
hold off

ylabel('RMSE', 'fontsize', 14)


lgd = legend('Monotone-Concave Polynomial', 'Cobb-Douglas');
set(lgd, 'Location', 'Best')
%set(gca,'xticklabel',labels,'fontsize',9)
%xtickangle(50)
set(gca,'yticklabel',num2str(abs(get(gca,'ytick').')))

name = "comp_RMSE_test";
fig_name = name + '.fig';
png_name = name + '.png';
savefig(fig_name);
saveas(fig,png_name,'png');


%% Compute competitive ratio
comp_ratio_test = cd_rmse_test_arr./mc_rmse_test_arr;
comp_ratio_train = cd_rmse_train_arr./mc_rmse_train_arr;

[comp_ratio_test,sortIdx] = sort(comp_ratio_test,'descend');
comp_ratio_train = comp_ratio_train(sortIdx);

% sort B using the sorting index
labels = industry_codes.Description(sortIdx);
% remove nans
nanIdx = ~isnan(comp_ratio_test) & ~isnan(comp_ratio_train);
comp_ratio_test = comp_ratio_test(nanIdx);
comp_ratio_train = comp_ratio_train(nanIdx);
% combine the ratio in a matrix
comp_ratio = [comp_ratio_test, comp_ratio_train];
labels = labels(nanIdx);

fig = figure();
c_map = lines(3);
hold on
b1 = bar(comp_ratio_test,'stack');
b1.FaceColor = c_map(1,:);
b1.EdgeAlpha = 0;
b2 = bar(-comp_ratio_train,'stack');
b2.FaceColor = c_map(3,:);
b2.EdgeAlpha = 0;
hold off

ylabel('RMSE competitive ratio', 'fontsize', 14)
yline(1,'-k','', 'Color', c_map(2,:), 'LineWidth', 2)
yline(-1,'-k','', 'Color', c_map(2,:), 'LineWidth', 2)

set(gca,'yticklabel',num2str(abs(get(gca,'ytick').')))

lgd = legend('Test ratio', 'Train ratio', 'Ratio = 1');
set(lgd, 'Location', 'Best')
%set(gca,'xticklabel',labels,'fontsize',9)
%xtickangle(50)

name = "comp_ratio_test_train";
fig_name = name + '.fig';
png_name = name + '.png';
savefig(fig_name);
saveas(fig,png_name,'png');

%% Plot just the test:
fig = figure();
b = bar(comp_ratio_test);
b.FaceColor = c_map(1,:);
b.EdgeAlpha = 0;


ylabel('RMSE competitive ratio', 'fontsize', 14)
yline(1,'-k','', 'Color', c_map(2,:), 'LineWidth', 2)
n_winners = sum(comp_ratio_test > 1);
xline(n_winners+0.5,'-','.', 'Color', c_map(3,:), 'LineWidth', 1)

lgd = legend('Test ratio', 'Ratio = 1');
set(lgd, 'Location', 'Best')

midpoint = 0.719;
annotation('textarrow',[.3,midpoint],...
           [.5,.5],'String','50 Industries ', 'LineWidth', 1)
annotation('textarrow',[0.78,midpoint,],...
           [.5,.5],'String','15 Industries', 'LineWidth', 1)       

name = "comp_ratio_test";
fig_name = name + '.fig';
png_name = name + '.png';
savefig(fig_name);
saveas(fig,png_name,'png');









%% Plot 3D
[X,Y] = meshgrid(min(features(:,1)):10000:max(features(:,1)), ...
                 min(features(:,2)):10000:max(features(:,2)));
c = mean(features(:,3));
data = [X(:), Y(:), c*ones(length(X(:)), 1)];
Z = exp(mdl.predict(log(data)));
Z = reshape(Z, size(X));
s = surf(X,Y,Z,'FaceAlpha',0.4, 'FaceColor','b');
hold on
% poly function
Y_hat = zeros(length(data), 1);
for j = 1:length(Y_hat)
    Y_hat(j) = replace(p, x, data(j,:)/100000);
end
Z = 100000*Y_hat;
Z = reshape(Z, size(X));
s = surf(X,Y,Z,'FaceAlpha',0.4, 'FaceColor','r');
hold off
legend('Cobb-Douglas','Monotone Convex')

%% Plot 2D
X = min(features(:,1)):10000:max(features(:,1));
c_1 = mean(features(:,2));
c_2 = mean(features(:,3));
N = length(X);
data = [X', c_1*ones(N,1), c_2*ones(N,1)];
Y = exp(mdl.predict(log(data)));
plot(X,Y, 'LineWidth', 2, 'color', 'b')
hold on
% poly function
Y_hat = zeros(length(data), 1);
for j = 1:length(Y_hat)
    Y_hat(j) = replace(p, x, data(j,:)/100000);
end
Z = 100000*Y_hat;
plot(X,Z, 'LineWidth', 2, 'color', 'r')

scatter(features(:,1), response)
hold off
legend('Cobb-Douglas','Monotone Convex', 'Data')

%% Plot contour

[X,Y] = meshgrid(min(features(:,1)):10000:max(features(:,1)), ...
                 min(features(:,2)):10000:max(features(:,2)));
c = mean(features(:,3));
data = [X(:), Y(:), c*ones(length(X(:)), 1)];
Z = exp(mdl.predict(log(data)));
Z = reshape(Z, size(X));
c = contour(X,Y,Z);
hold on
% poly function
Y_hat = zeros(length(data), 1);
for j = 1:length(Y_hat)
    Y_hat(j) = replace(p, x, data(j,:)/100000);
end
Z = 100000*Y_hat;
Z = reshape(Z, size(X));
c = contour(X,Y,Z);
hold off
legend('Cobb-Douglas','Monotone Convex')