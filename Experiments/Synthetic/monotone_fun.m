function f = monotone_fun(arg)
%% Description
% Outputs
%   f := a column vector of the evaluations of the function
% Inputs
%   arg := the input of the function 

[~, k] = size(arg); 
% f=10./(1+1*exp((-1*sum(arg,2)+1.25*k)));
f = 5 ./ (1 + exp(-5*k*sum((arg - mean(arg)),2)));
end