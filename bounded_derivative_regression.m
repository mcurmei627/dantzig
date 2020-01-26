function [p,x] = ...
    bounded_derivative_regression(degree,features,response,l_bound,u_bound)
%% Description 
% Outputs
%   p := decision polynomial
%   x := argument of p
% Inputs
%   degree := the degree  of the decision polynomial 
%   features := feature variable data used for training
%   response := response variable used for training
%   l_bound := real valued vector such that each entry correcponds to the
%              lower bound on the partial derivative
%   u_bound := real valued vector such that each entry correcponds to the
%              upper bound on the partial derivative

%% Clear up the environment
% This is an important step for improving performance
yalmip('clear');
%% PROBLEM SETUP: Define the box
% find the superior and inferior bounds 
% (currently we infer the bounds based on the full datasets, 
% in actual applications rescalling is necessary due to 
% numerical problems
tol = 0.001;
inf_domain = min(features) - tol;
sup_domain = max(features) + tol;

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

monom_bulk = bulkeval(v,x,features');
peval_bulk = c'*monom_bulk;
diff_bulk = peval_bulk - response';
h = diff_bulk*diff_bulk';

%% PROBLEM SETUP: Define the decision variables used in the constraints

% Create the monomials used in the constraints
monomials = monolist(x, degree-2); % degree-2 because the other decision polynomials have degree degreemono-2

% Define the coefficients of the matrices of helper polynomials
l_coef_help = sdpvar(k*k, length(monomials));
u_coef_help = sdpvar(k*k, length(monomials));

% Create the matrix of helper polynomials 
l_Q_help = l_coef_help*monomials;
l_Q_help = reshape(l_Q_help, k, k, []);

u_Q_help = u_coef_help*monomials;
u_Q_help = reshape(u_Q_help, k, k, []);

%% PROBLEM SETUP: Write the constraints
F = [sos(l_Q_help), sos(u_Q_help)];
F = F+[sos(transpose(jacobian(p,x))- l_bound - l_Q_help*transpose((x-inf_domain).*(sup_domain-x)))];
F = F+[sos(u_bound-transpose(jacobian(p,x))- u_Q_help*transpose((x-inf_domain).*(sup_domain-x)))];
%% SOS OPTIMIZATION: Fit the desired polynomial
options = sdpsettings('verbose',2, 'solver', 'mosek');
all_coef = [c;reshape(l_coef_help, k*k*length(monomials),1,[]);reshape(u_coef_help, k*k*length(monomials),1,[])];
[sol,m,B,residual]=solvesos(F, h, options, all_coef);

%% Display message
msg = "Bounded derivative regression for polynomial of degree "+degree+" complete.";
disp(msg);
end