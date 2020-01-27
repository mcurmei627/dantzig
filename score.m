function [err_train,err_test] = score(algo,degree,features_train,...
                 response_train,features_test, response_test, varargin)
%% Description
% Outputs
%   err_train := MSE on the training dataset
%   err_test := MSE on the testing dataset
% Inputs
%   algo := the regression algoritm - monotone, convex or unconstrained
%   degree := degree of the polynomial
%   features_train := feature variable data used for training
%   response_train :=  response variable used for training
%   features_test := feature variable data used for testing
%   response_test := response variable used for testing
%   varargin := optional argument; a sequence of the form:(or any subset of it)
%                      'monotone_profile', ones(3,1),
%                      'convex_sign', -1,
%                      'l_bound', [-1, 2],
%                      'u_bound', [3, 5]

%% Parse the input
[~, k] = size(features_train);  
N_train = length(response_train);
N_test = length(response_test);

%  Define the acceptable names for helper variables in the optional
%  varargin
arg_struct = struct('monotone_profile', ones(k,1),...
                    'convex_sign', 1, ...
                    'l_bound', -ones(k,1),...
                    'u_bound', ones(k,1));
arg_names = fieldnames(arg_struct);

% Count arguments and ensure that they come in pairs
n_args = length(varargin);
if round(n_args/2)~=n_args/2
   error('Property names and property values must come in pairs')
end

% Populate the optional arguments struct with user defined values
for pair = reshape(varargin,2,[]) % pair is {arg; value}
    arg = pair{1}; 

   if any(strcmp(arg, arg_names))
      arg_struct.(arg) = pair{2};
   else
      error('%s is not a recognized parameter name',arg)
   end
end

%% Fit the desired model
switch algo
    case "monotone"
        % fit to the training data
        [p,x] = monotone_regression(degree,...
                                    features_train,response_train,...
                                    arg_struct.monotone_profile);
    case "bounded_derivative"
        % fit to the training data
        [p,x] = bounded_derivative_regression(degree,...
                                              features_train,...
                                              response_train,...
                                              arg_struct.l_bound,...
                                              arg_struct.u_bound);                                       
    case "convex"
        % fit to the training data
        [p,x] = convex_regression(degree,...
                                  features_train,response_train,...
                                  arg_struct.convex_sign);   
    case "monotone_convex"
        % fit to the training data
        [p,x] = monotone_convex_regression(degree,...
                                           features_train,response_train,...
                                           arg_struct.monotone_profile, ...
                                           arg_struct.convex_sign);            
end

%% Compute the predictions
switch algo   
    case {"monotone", "bounded_derivative", "convex", "monotone_convex"}
        % compute predicted responses for training and testing dataset
        Y_hat_train = repmat(0, [N_train 1]);
        for i = 1:N_train
            Y_hat_train(i) = replace(p, x, features_train(i,:));
        end
        Y_hat_test = repmat(0, [N_test 1]);
        for i = 1:N_test
            Y_hat_test(i) = replace(p, x, features_test(i,:));
        end     
                                     
    case "unconstrained"
        mdl = unconstrained_regression(degree,features_train,response_train);
        Y_hat_train = mdl.predict(features_train);
        Y_hat_test = mdl.predict(features_test);
        
    otherwise
        msg="Error: " + algo + " not found, choose between 'monotone', " + ...
            "'bounded_derivative', 'convex', 'monotone_convex' or 'unconstrained'";
        error(msg);
end

%% Compute the scores   
% Average squared deviation in the training set
err_train = value(transpose(Y_hat_train-response_train)*...
                  (Y_hat_train-response_train))/N_train;
% Average squared deviation in the testing set
err_test = value(transpose(Y_hat_test-response_test)*...
                 (Y_hat_test-response_test))/N_test;
end
        
        
        
                    