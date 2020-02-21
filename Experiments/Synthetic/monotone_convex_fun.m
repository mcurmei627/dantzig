function f = monotone_convex_fun(arg)
%% Description
% Outputs
%   f := a column vector of the evaluations of the function
% Inputs
%   arg := the input of the function 
f=log_sum_exp(arg, 2).^2;
end