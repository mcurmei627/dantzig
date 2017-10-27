function f = expo_fun(arg)
%% Description
% Outputs
%   f := a column vector of the evaluations of the function
% Inputs
%   arg := the input of the function 
[~, k] = size(arg);
arg_af=(sum(arg,2)-0.5*k)/(1.5*k);
f=(arg_af.*log(arg_af))*10+5;
end