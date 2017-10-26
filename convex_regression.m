function [p,x] = convex_regression(degree,features,response,convex_sign)
%% Description 
% Outputs
%   p := decision polynomial
%   x := argument of p
% Inputs
%   degree := the degree  of the decision polynomial 
%   features := feature variable data used for training
%   response := response variable used for training
%   convex_sign := 1 if convex, -1 if concave

%% Clear up the environment
% This is an important step for improving performance
yalmip('clear');
%% PROBLEM SETUP: Define the box
% find the superior and inferior bounds 
% (currently we infer the bounds based on the full datasets, 
% in actual applications rescalling is necessary due to 
% numerical problems
tol = 0.001;
inf_bound = min(features) - tol;
sup_bound = max(features) + tol;

%% PROBLEM SETUP: Define the parameters and decision variables

%k - number of features
[~, k] = size(features); 
x=sdpvar(1,k);

% Define the main polynomial to be learned
% p is the polynomial
% c is the array of the cofficients of the polynomial p
% v is the array of monomials
[p,c,v] = polynomial(x,degree);

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
h = diff_bulk*diff_bulk'; % <- h is the minimization objective, the sum of 
                          %    squared errors

%% PROBLEM SETUP: Define the decision variables used in the constraints

% Create helper free variable
y=sdpvar(1,k);

% Create the monomials of the helper polynomials used in the constraints
monomials = monolist([x y], degree-2);

% Define the coefficients of the array of helper polynomials
coef_help = sdpvar(k, length(monomials));

% Create an array of helper polynomials 
Q_help = coef_help*monomials;

%% PROBLEM SETUP: Write the constraints
F = [sos(Q_help)];
F = F+[sos(y*hessian(p,x)*transpose(y).*convex_sign-(x-inf_bound).*(sup_bound-x)*Q_help)];

%% SOS OPTIMIZATION: Fit the desired polynomial
options = sdpsettings('verbose',2, 'solver', 'mosek');
% The coefficients are the decision variables, putting them all in an array
all_coef = [c;reshape(coef_help, k*length(monomials),1,[])];
[sol,m,B,residual]=solvesos(F, h, options, all_coef);
end
