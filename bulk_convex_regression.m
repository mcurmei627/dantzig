function [p,x, exp_output] = bulk_convex_regression(degree,helper_degrees, features,response)
%% Description 
% Outputs
%   p := decision polynomial
%   x := argument of p
% Inputs
%   degree := the degree  of the decision polynomial 
%   features := feature variable data used for training
%   response := response variable used for training
%   convex_sign := 1 if convex, -1 if concave
%   varargin: optional argument; a sequence of the form:(or any subset of it)
%                      'solver', 'mosek',
%                      'helper_degree', 2


%% Clear up the environment
% This is an important step for improving performance
yalmip('clear');
options = sdpsettings('verbose',0, 'solver', 'mosek');

%% 
exp_output = struct('solver_time' , {}, ...
    'train_rmse', {}, ...
    'test_rmse',{},...
    'N', {}, ...
    'degree', {}, ...
    'helper_degree', {}, ...
    'features' , {});
%% PROBLEM SETUP: Define the box
tol = 0.001;
inf_domain = min(features) - tol;
sup_domain = max(features) + tol;

%% PROBLEM SETUP: Define the parameters and decision variables
t0 = tic();
[~, k] = size(features); 
x=sdpvar(1,k);
[p,c,v] = polynomial(x,degree);

% valid_index = [1:500, 10000:10050];
% features = features(valid_index, :);
% response = response(valid_index);
%% PROBLEM SETUP: Write the objective
monom_bulk = bulkeval(v,x,features'); % <- evaluates all monomials for all values of the features
peval_bulk = c'*monom_bulk;
diff_bulk = peval_bulk - response'; 
setup_time = toc(t0);
msg = "Setup time: " + setup_time + " seconds.";
disp(msg);

n_test = 1000;
test_diff_vector =  diff_bulk(end-n_test+1:end);
test_rmse = norm(test_diff_vector, 2);
%% PROBLEM SETUP: Define the decision variables used in the constraints
% Iterate through helper_degree:
y=sdpvar(1,k);
index = 1;
for helper_degree = helper_degrees
    mono_degree = cat(2, repelem(helper_degree, k), repelem(2, k));

    monomials = monolist([x y], mono_degree);

    % Define the coefficients of the array of helper polynomials
    coef_help = sdpvar(k, length(monomials));

    % Create an array of helper polynomials 
    Q_help = coef_help*monomials;

    %% PROBLEM SETUP: Write the constraints
    F = [sos(Q_help)];
    F = F+[sos(y*hessian(p,x)*transpose(y)-(x-inf_domain).*(sup_domain-x)*Q_help)];
    all_coef = [c;reshape(coef_help, k*length(monomials),1,[])];
    %% SOS OPTIMIZATION: Fit the desired polynomial
    for N = [2000, 5000, 10000]
        h = norm(diff_bulk(1:N),2);
        t1 = tic();
        [sol,m,B,residual]=solvesos(F, h, options, all_coef);
        optimization_time = toc(t1);
        msg = "Optimization runtime: " + optimization_time + " seconds. (N = " + N + " helper_degree = " + helper_degree + ")";
        display(msg);
        train_rmse = value(h)/sqrt(N);
        test_rmse_val = value(test_rmse)/sqrt(n_test);
        aux_out = struct( 'solver_time', sol.('solvertime'), 'train_rmse', train_rmse,...
        'test_rmse', test_rmse_val, 'N', N, 'degree', degree, 'helper_degree', helper_degree, 'features', k);
        exp_output(index) = aux_out;
        display(aux_out);
        index = index +1;
        clear h;
    end
    clear F all_coef Q_help monomials  
    %% Display message
    msg = "Convex regression for polynomial of degree "+degree+ ...
        "and helper degree " + helper_degree + " complete.";
    disp(msg);
end


