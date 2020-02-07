function [train_uncs,test_uncs,train_mcpr,test_mcpr]=...
    credit_regression_cv(degree)
%% Read the csv file with the education data
filename_features = fullfile(pwd, 'Data', 'master_house_features.csv');
filename_response = fullfile(pwd, 'Data', 'master_house_response.csv');

features = csvread(filename_features);
response = csvread(filename_response);

[N,k] = size(features);

train_uncs = zeros(10,1);
train_mcpr = zeros(10,1);
test_uncs = zeros(10,1);
test_mcpr = zeros(10,1);

monotone_profile = [1; 1; 1; 1; -1; -1];
%% Split the  data set into training and testing
% Split into test and train (note that we do not have a validation set)
idx = crossvalind('Kfold', 100, 10);
%%
for i = 1:10
    idx_test = (idx == i); idx_train = ~idx_test;
    
    features_train = features(idx_train,:);
    features_test = features(idx_test,:);

    response_train = response(idx_train);
    response_test = response(idx_test);
    
    %% Rescale the features
    % make sure testing data undergoes the same afine transformation as the
    % training data
    features_test = (features_test- min(features_train))./(max(features_train)-min(features_train))*(1.5-0.5) + 0.5;
    features_train = (features_train- min(features_train))./(max(features_train)-min(features_train))*(1.5-0.5) + 0.5;

    %% Perform experiment from unconstrained and monotonic regression
    [train_uncs(i),test_uncs(i)]=score('unconstrained',degree-1,features_train,...
                 response_train,features_test, response_test,'','');
    [train_mcpr(i),test_mcpr(i)]=score('monotone',degree,features_train,...
                 response_train,features_test, response_test,monotone_profile,'');
end