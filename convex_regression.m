function [p,x, aux_out] = convex_regression(d,features,response,convex_sign, varargin)
%% Description 
% Outputs
%   p := decision polynomial
%   x := argument of p
% Inputs
%   d := the degree  of the decision polynomial 
%   features := feature variable data used for training
%   response := response variable used for training
%   convex_sign := 1 if convex, -1 if concave
%   varargin: optional argument; a sequence of the form:(or any subset of it)
%                      'solver', 'mosek',
%                      'helper_degree', 2


%% Clear up the environment
% This is an important step for improving performance
yalmip('clear');
%% Define the acceptable names for helper variables in the optional
%  varargin
arg_struct = struct('solver', 'mosek', 'helper_degree', d-2);
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
%% PROBLEM SETUP: Define the box
% find the superior and inferior bounds 
% (currently we infer the bounds based on the full datasets, 
% in actual applications rescalling is necessary due to 
% numerical problems
tol = 0.001;
inf_domain = min(features) - tol;
sup_domain = max(features) + tol;

%% PROBLEM SETUP: Define the parameters and decision variables
t0 = tic();
%k - number of features
[N, k] = size(features); 
x=sdpvar(1,k);

% Define the main polynomial to be learned
% p is the polynomial
% c is the array of the cofficients of the polynomial p
% v is the array of monomials
[p,c,v] = polynomial(x,d);

%% PROBLEM SETUP: Write the objective
% currently computing the objective is the biggest computational bottleneck
% I tried to vectorized versions, but they lead to errors

monom_bulk = bulkeval(v,x,features'); % <- evaluates all monomials for all values of the features
% >> monom_bulk
% monom_bulk =
%   Columns 1 through 12
%     1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000
%     1.2557    1.2344    1.8156    1.0297    1.1742    1.9453    0.5634    1.9594    0.7838    1.5007    1.3797    1.5127
%     1.1101    1.5004    1.9006    1.7164    ... 

peval_bulk = c'*monom_bulk;% <- computes the value of the polynomial given the value of
                           %    the argument from future, as a function of
                           %    polynomial coefficients c
% >> display(peval_bulk)
%   Linear matrix variable 1x100 (full, real, 210 variables)
% >> sdisplay(peval_bulk(1,1))
%   c(1)+1.25567117866*c(2)+1.11009037263*c(3)+1.25576012277*c(4)+0.710383038904*c(5)+...

diff_bulk = peval_bulk - response'; % <- computes the difference between
                                    %    the function value at the feature
                                    %    input and the response, (as a
                                    %    function of c)
%h = diff_bulk*diff_bulk'; % <- h is the minimization objective, the sum of 
                          %    squared errors
h = norm(diff_bulk,2);
%% PROBLEM SETUP: Define the decision variables used in the constraints

% Create helper free variable
y=sdpvar(1,k);

% Create the monomials of the helper polynomials used in the constraints
r = arg_struct.('helper_degree');
mono_degree = cat(2, repelem(2*r-2, k), repelem(2, k));
% (the max degree associated with the helper variable is 2)
monomials = monolist([x y], mono_degree);

% Define the coefficients of the array of helper polynomials
coef_help = sdpvar(k, length(monomials));

% Create an array of helper polynomials 
Q_help = coef_help*monomials;

%% PROBLEM SETUP: Write the constraints
F = [sos(Q_help)];
F = F+[sos(y*hessian(p,x)*transpose(y).*convex_sign-(x-inf_domain).*(sup_domain-x)*Q_help)];

%% SOS OPTIMIZATION: Fit the desired polynomial
options = sdpsettings('verbose',0, 'solver', arg_struct.('solver'));
% The coefficients are the decision variables, putting them all in an array
all_coef = [c;reshape(coef_help, k*length(monomials),1,[])];
setup_time = toc(t0);
msg = "Setup time: " + setup_time + " seconds.";
disp(msg);
t1 = tic();
[sol,m,B,residual]=solvesos(F, h, options, all_coef);
optimization_time = toc(t1);
msg = "Optimization runtime: " + optimization_time + " seconds.";
%disp(msg);

%% Display message
msg = "Convex regression for polynomial of degree "+d+ ...
    "and helper degree " + arg_struct.('helper_degree') + " complete.";
disp(msg);

% get the min eigen value
l = length(B);
current_monomials = m{l};
keep_idx = [];
for i = 1:length(current_monomials)
    degree_x = sum(degree(current_monomials(i), x));
    degree_y = sum(degree(current_monomials(i), y));
    if degree_y == 1
        if degree_x <= r
            keep_idx = [keep_idx, i];
        end
    end
end

gram_matrix = B{l};
gram_matrix = gram_matrix(keep_idx, keep_idx);

aux_out = struct('setup_time', setup_time, 'optimization_time',...
    optimization_time, 'solver_time', sol.('solvertime'), ...
    'train_rmse', sqrt(value(h)^2/N), 'Q', gram_matrix);
end
