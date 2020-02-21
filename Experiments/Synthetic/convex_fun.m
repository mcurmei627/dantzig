function f = convex_fun(arg)
%% Description
% Outputs
%   f := a column vector of the evaluations of the function
% Inputs
%   arg := the input of the function 

[~, k] = size(arg);
arg_af=(abs(sum(arg,2)-0.7*k))/(1.5*k);
f=(arg_af.*log(arg_af))*30+5;
% plot ((x + y) - 1)/3 * log( ((x + y) - 1)/3)  for x, y = [0.5, 2]

% f = (vecnorm(arg,2,2));
end

