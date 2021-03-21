function [f,x, alpha, grad_fx] = get_monge_potential(P_mat, X_mat, Y_mat, degree, helper_degree, ...
    small_ell, big_ell, inf_domain, sup_domain)
yalmip('clear');
[n, k] = size(X_mat); 
[m, ~] = size(Y_mat);
x=sdpvar(1,k);
[f,c,v] = polynomial(x,degree);
grad_f = jacobian(f, x);
[grad_c,v] = coefficients(grad_f, x);
monom_bulk = bulkeval(v,x,X_mat');
grad_eval = grad_c*monom_bulk; % k by n matrix
X_terms = repmat(grad_eval',m ,1);
Y_terms = repelem(Y_mat', 1, n)';
diff = X_terms'-Y_terms';   % k x n*m matrix
stretched_P = repmat(reshape(sqrt(P_mat), 1, n*m), k, 1);
weighted_diff = diff.*stretched_P;
%objective = norm(weighted_diff);
wd = reshape(weighted_diff, 1, k*m*n);
objective = wd*wd';

%% add constraints
y=sdpvar(1,k);

% Create the monomials of the helper polynomials used in the constraints
mono_degree = cat(2, repelem(helper_degree, k), repelem(2, k));

% (the max degree associated with the helper variable is 2)
helper_v = monolist([x y], mono_degree);

coef_help = sdpvar(2*k, length(helper_v));
% Create an array of helper polynomials 
Q_help = coef_help*helper_v;

%% monotone
% m_monomials = monolist(x, helper_degree);
% 
% % Define the coefficients of the matrix of helper polynomials
% % for the MONOTONE constraints
% m_coef_help = sdpvar(2*k*k, length(m_monomials));
% 
% % Create the matrix of helper polynomials 
% m_Q_help = m_coef_help*m_monomials;
% m_Q_help = reshape(m_Q_help, 2*k, k, []);


%% PROBLEM SETUP: Write the constraints
F = [sos(Q_help)];
% F = F+[sos(transpose(jacobian(f,x)-inf_domain) - m_Q_help(1:k,:)*transpose((x-inf_domain).*(sup_domain-x)))];
% F = F+[sos(transpose(sup_domain - jacobian(f,x))- m_Q_help(k+1:2*k,:)*transpose((x-inf_domain).*(sup_domain-x)))];
% F = F+[sos(y*(hessian(f,x)-small_ell*eye(k))*transpose(y)- ...
%     (x-inf_domain)*Q_help(1:k) - (sup_domain-x)*Q_help(k+1:2*k))];
% F = F+[sos(y*(big_ell*eye(k)-hessian(f,x))*transpose(y)- ...
% (x-inf_domain)*Q_help(2*k+1:3*k) - (sup_domain-x)*Q_help(3*k+1:4*k))];
F = F+[sos(y*(hessian(f,x)-small_ell*eye(k))*transpose(y)- ...
    (x-inf_domain).*(sup_domain-x)*Q_help(1:k))];
F = F+[sos(y*(big_ell*eye(k)-hessian(f,x))*transpose(y)- ...
(x-inf_domain).*(sup_domain-x)*Q_help(k+1:2*k))];

options = sdpsettings('verbose',0, 'solver', 'mosek');
% The coefficients are the decision variables, putting them all in an array
all_coef = [c; reshape(grad_c, k*length(v),1,[]); ...
    reshape(coef_help, 2*k*length(helper_v),1,[])];
%reshape(m_coef_help, 2*k*k*length(m_monomials),1,[])];

[sol,~,B,residual]=solvesos(F, objective, options, all_coef);

alpha = reshape(vecnorm(value(diff)),n,m);
grad_fx = value(grad_eval');
disp(value(objective));
end