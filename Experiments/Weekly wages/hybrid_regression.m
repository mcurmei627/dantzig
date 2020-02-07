function [test_rmse,test_u_rmse,duration]=hybrid_regression(degree)
%% Description
% This function looks at the weekly wage data and regresses it with respect
% to two numerical variables, years of education and years of experience
% our assumption is that the wage is monotonically increasing with respect 
% to wage and concave with respect to years of experience.

%% Read the data
filename = fullfile(pwd, 'Data', 'usa_wk_mar_2017.csv');
data = csvread(filename,1,0); % offset by one row to skip the headers;
data = data( ismember(data(:,3), 10), :);
N = size(data,1);
k = 4;
response = data(:,8);
features = data(:,[9, 10, 11]);

%% Split the  data set into training and testing
% Split into test and train (note that we do not have a validation set)
[idx_train, ~, idx_test]  = dividerand(N, 0.8, 0, 0.2);

features_train = features(idx_train,:);
features_test = features(idx_test,:);

response_train = response(idx_train);
response_test = response(idx_test);

%% Rescale the features
% make sure testing data undergoes the same afine transformation as the
% training data
features_test = (features_test- min(features_train))./(max(features_train)-min(features_train))*(1.5-0.5) + 0.5;
features_train = (features_train- min(features_train))./(max(features_train)-min(features_train))*(1.5-0.5) + 0.5;
%% Develop hybrid model
%Clear up the environment
% This is an important step for improving performance
yalmip('clear');
%% PROBLEM SETUP: Define the box
% find the superior and inferior bounds 
% (currently we infer the bounds based on the full datasets, 
% in actual applications rescalling is necessary due to 
% numerical problems
tol = 0.001;
inf_bound = min(features_train) - tol;
sup_bound = max(features_train) + tol;

%% PROBLEM SETUP: Define the parameters and decision variables
x = sdpvar(1,k);
[p,c,v] = polynomial(x,degree);

%% PROBLEM SETUP: Write the objective
% currently computing the objective is the biggest computational bottleneck
% I tried to vectorized versions, but they lead to errors

monom_bulk = bulkeval(v,x,features_train');
peval_bulk = c'*monom_bulk;
diff_bulk = peval_bulk - response_train'; 
h = diff_bulk*diff_bulk'; % <- h is the minimization objective, the sum of 
                          %    squared errors

%% PROBLEM SETUP: Define the decision variables used in the monotone constraint
tic
% Create the monomials used in the constraints
monomials = monolist(x, degree-2); 

% Define the coefficients of the matrix of helper polynomials for the
% monotone constraint
coef_monotone = sdpvar(k*k, length(monomials));

% Create the matrix of helper polynomials for the monotone constraint
Q_monotone = coef_monotone*monomials;
Q_monotone = reshape(Q_monotone, k, k, []);

% Monotone profile
monotone_profile = [1; 0];
%% PROBLEM SETUP: Define the decision variables used in the convex constraint
% Define the coefficients of the matrix of helper polynomials for the
% monotone constraint
coef_convex = sdpvar(k*k, length(monomials));

% Create the matrix of helper polynomials for the convex constraint
Q_convex = coef_convex*monomials;
Q_convex = reshape(Q_convex, k, k, []);

% Convex profile
convex_profile = [0;-1];
%% PROBLEM SETUP: Write the constraints
F = [sos(Q_monotone)];
F = F+[sos(Q_convex)];
F = F+[sos(transpose(jacobian(p,x)).*monotone_profile -  Q_monotone*transpose((x-inf_bound).*(sup_bound-x)))];
F = F+[sos(diag(hessian(p,x)).*convex_profile -  Q_convex*transpose((x-inf_bound).*(sup_bound-x)))];

%% SOS OPTIMIZATION: Fit the desired polynomial
options = sdpsettings('verbose',0, 'solver', 'mosek');
all_coef = [c;reshape(coef_monotone, k*k*length(monomials),1,[]);...
              reshape(coef_convex, k*k*length(monomials),1,[])];

[sol,m,B,residual]=solvesos(F, h, options, all_coef);
duration=toc

%% Fit unconstrained polynomial regression and get the score
[err_train_uncs,err_test_uncs] = score('unconstrained',degree,...
            features_train,response_train,features_test, response_test);
train_u_rmse = sqrt(err_train_uncs);        
test_u_rmse = sqrt(err_test_uncs);  
%%
N_train = length(response_train);
N_test = length(response_test);             
% fit to the training data
% get the coefficients
coef = coefficients(p,x);
% compute predicted responses for training and testing dataset
Y_hat_train = repmat(0, [N_train 1]);
for i = 1:N_train
    Y_hat_train(i) = replace(p, x, features_train(i,:));
end
Y_hat_test = repmat(0, [N_test 1]);
for i = 1:N_test
    Y_hat_test(i) = replace(p, x, features_test(i,:));
end

% Average squared deviation in the training set
err_train = value(transpose(Y_hat_train-response_train)*(Y_hat_train-response_train))/N_train;
% Average squared deviation in the testing set
err_test = value(transpose(Y_hat_test-response_test)*(Y_hat_test-response_test))/N_test;

train_rmse = sqrt(err_train);        
test_rmse = sqrt(err_test);  


end