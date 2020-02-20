function [p,x] = monotone_regression(degree,features,response,monotone_profile)
%% Description 
% Outputs
%   p := decision polynomial
%   x := argument of p
% Inputs
%   degree := the degree  of the decision polynomial 
%   features := feature variable data used for training
%   response := response variable used for training
%   monotone_profile := the vector of 0,1,-1, describing the monotonicity
%                       relationship between each feature and the response

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

monom_bulk = bulkeval(v,x,features');
peval_bulk = c'*monom_bulk;
diff_bulk = peval_bulk - response';
h = diff_bulk*diff_bulk';

%% PROBLEM SETUP: Define the decision variables used in the constraints

% Create the monomials used in the constraints
monomials = monolist(x, degree-2); % degree-2 because the other decision 
                                   % polynomials have degree-2

% Define the coefficients of the matrix of helper polynomials
coef_help = sdpvar(k*k, length(monomials));

% Create the matrix of helper polynomials 
Q_help = coef_help*monomials;
Q_help = reshape(Q_help, k, k, []);

%% PROBLEM SETUP: Write the constraints
F = [sos(Q_help)];
F = F+[sos(transpose(jacobian(p,x)).*monotone_profile - ...
       Q_help*transpose((x-inf_bound).*(sup_bound-x)))];

%% SOS OPTIMIZATION: Fit the desired polynomial
options = sdpsettings('verbose',0, 'solver', 'mosek');
all_coef = [c;reshape(coef_help, k*k*length(monomials),1,[])];
[sol,m,B,residual]=solvesos(F, h, options, all_coef);

%% Display message
msg = "Monotone regression for polynomial of degree "+degree+" complete.";
disp(msg);
end