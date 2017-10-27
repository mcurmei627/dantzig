function [err_train,err_test] = score(algo,degree,features_train,...
                 response_train,features_test, response_test)
%% Description
% Outputs
%   err_train := MSE on the training dataset
%   err_test := MSE on the testing dataset
% Inputs
%   algo := the regression algoritm - monotone, convex or unconstrained
%   degree := degree of the polynomial
%   features_train
%   response_train
%   features_test
%   response_test
%%
[~, k] = size(features_train);  
N_train = length(response_train);
N_test = length(response_test);

switch algo
    case "monotone"
        % fit to the training data
        [p,x] = monotone_regression(degree,features_train,response_train,ones(k,1));
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
        
        
    case "convex"
        % fit to the training data
        [p,x] = convex_regression(degree,features_train,response_train,ones(k,1)); 
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
        
        
    case "unconstrained"
        mdl = unconstrained_regression(degree,features_train,response_train);
        Y_hat_train = mdl.predict(features_train);
        Y_hat_test = mdl.predict(features_test);
        
    otherwise
        msg="Error: "+algo+" not found, choose between 'monotone', 'convex' or 'unconstrained'";
        error(msg);
end

% Average squared deviation in the training set
err_train = value(transpose(Y_hat_train-response_train)*(Y_hat_train-response_train))/N_train;
% Average squared deviation in the testing set
err_test = value(transpose(Y_hat_test-response_test)*(Y_hat_test-response_test))/N_test;
end
        
        
        
                    