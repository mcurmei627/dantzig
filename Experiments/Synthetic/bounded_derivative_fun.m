function f = bounded_derivative_fun(arg)
%% Description
% Outputs
%   f := a column vector of the evaluations of the function
% Inputs
%   arg := the input of the function 

f = (vecnorm(arg,2,2));
end