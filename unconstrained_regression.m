function mdl = unconstrained_regression(degree,features,response)
%% Unconstrained polynomial regression
% Outputs
%   mdl := model from fitlm customized for multivariate polynomial regression
% Inputs
%   degree := the degree  of the decision polynomial 
%   features := feature variable data used for training
%   response := response variable used for training

%%
%k - number of features
[~, k] = size(features); 

% this is a trick to make fitlm fit a polynomial, for example if
% mdl_spec is 'poly666' then fitml will fit a polynomial of degree 6 in the
% first 3 features
mdl_spec = strcat('poly', repmat(int2str(degree),1, k));

%%
mdl = fitlm(features,response',mdl_spec);
%% Display message
msg = "Unconstrained regression for polynomial of degree "+degree+" complete.";
disp(msg);
end