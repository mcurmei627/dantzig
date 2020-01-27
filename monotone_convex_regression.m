function [p,x] = monotone_convex_regression(degree,features,response,monotone_profile,convex_sign)
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

%% Monotone
% Create the monomials of the helper polynomials used in the 
% MONOTONE constraints
m_monomials = monolist(x, degree-2);

% Define the coefficients of the matrix of helper polynomials
% for the MONOTONE constraints
m_coef_help = sdpvar(k*k, length(m_monomials));

% Create the matrix of helper polynomials 
m_Q_help = m_coef_help*m_monomials;
m_Q_help = reshape(m_Q_help, k, k, []);

%% Convex
% Create helper free variable for the CONVEX constraints
y=sdpvar(1,k);

% Create the monomials of the helper polynomials used in the constraints
mono_degree = cat(2, repelem(degree-2, k), repelem(2, k));
% (the max degree associated with the helper variable is 2)
c_monomials = monolist([x y], mono_degree);

% Define the coefficients of the array of helper polynomials
% for the CONVEX constraint
c_coef_help = sdpvar(k, length(c_monomials));

% Create an array of helper polynomials 
c_Q_help = c_coef_help*c_monomials;

%% PROBLEM SETUP: Write the constraints
F = [sos(m_Q_help), sos(m_Q_help)];
% Add monotonicity constraints
F = F+[sos(transpose(jacobian(p,x)).*monotone_profile - m_Q_help*transpose((x-inf_domain).*(sup_domain-x)))];
% Add convexity constraints
F = F+[sos(y*hessian(p,x)*transpose(y).*convex_sign-(x-inf_domain).*(sup_domain-x)*c_Q_help)];

%% SOS OPTIMIZATION: Fit the desired polynomial
options = sdpsettings('verbose',2, 'solver', 'mosek');
% The coefficients are the decision variables, putting them all in an array
all_coef = [c;reshape(c_coef_help, k*length(c_monomials),1,[]);reshape(m_coef_help, k*k*length(m_monomials),1,[])];
[sol,m,B,residual]=solvesos(F, h, options, all_coef);

%% Display message
msg = "Monotone-convex regression for polynomial of degree "+degree+" complete.";
disp(msg);
end
