N_degrees = 5;
train_uncs_mat = zeros(10,N_degrees);
train_mcpr_mat = zeros(10,N_degrees);
test_uncs_mat = zeros(10,N_degrees);
test_mcpr_mat = zeros(10,N_degrees);

for d=1:N_degrees
    [train_uncs_mat(:,d),test_uncs_mat(:,d),train_mcpr_mat(:,d),...
        test_mcpr_mat(:,d)]=credit_regression_cv(d);
end

%% combine the columns for train and test
train_mat = [train_uncs_mat,train_mcpr_mat];
test_mat= [test_uncs_mat,test_mcpr_mat];
%% Training data
fig = figure()
bottom_label = [repmat({'UPR'},1,N_degrees),repmat({'MCPR'},1,N_degrees)];
top_label = repmat({'            d=1',...
                    '            d=2',...
                    '            d=3',...
                    '            d=4',...
                    '            d=5'},1,2);
h=boxplot(train_mat,{top_label,bottom_label},'colors',repmat([0, 0.6, 0.6;0.4 0 0.4],N_degrees,1),...
    'factorgap', [4,2],'labelverbosity','minor','Widths',0.4,...
    'FactorSeparator',[1],'OutlierSize',4);
set(h,'LineWidth',1.2)
title(['RMSE on the training data']); 
ylabel('RMSE')
grid on

% save the figure and a PNG plot 
fig_name = 'train_rmse_credit.fig';
wd = pwd;
plot_dir = fullfile(wd, 'Plots');
cd(plot_dir);
savefig(fig_name);
cd(wd)

%% Testing data
fig = figure()
bottom_label = [repmat({'UPR'},1,N_degrees),repmat({'MCPR'},1,N_degrees)];
top_label = repmat({'            d=1',...
                    '            d=2',...
                    '            d=3',...
                    '            d=4',...
                    '            d=5'},1,2);
h=boxplot(test_mat,{top_label,bottom_label},'colors',repmat([0, 0.6, 0.6;0.4 0 0.4],N_degrees,1),...
    'factorgap', [4,2],'labelverbosity','minor','Widths',0.4,...
    'FactorSeparator',[1],'OutlierSize',4, 'DataLim', [0, 15]);
set(h,'LineWidth',1.2)
title(['RMSE on the testing data']); 
ylabel('RMSE')
grid on

% save the figure and a PNG plot 
fig_name = 'test_rmse_credit.fig';
wd = pwd;
plot_dir = fullfile(wd, 'Plots');
cd(plot_dir);
savefig(fig_name);
cd(wd)