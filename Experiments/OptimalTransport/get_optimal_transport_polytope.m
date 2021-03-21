function P_mat = get_optimal_transport_polytope(alpha_mat, a_vec, b_vec)

yalmip('clear')
a_vec = reshape(a_vec, [],1);
b_vec = reshape(b_vec, [],1);
[n, m] = size(alpha_mat);
P = sdpvar(n, m, 'full');

options = sdpsettings('verbose',0, 'solver', 'mosek');
objective = sum(sum(P.*alpha_mat));
constraints = [P>=0, P*ones(m,1)==a_vec, ones(1, n)*P==b_vec'];
optimize(constraints, objective, options);

P_mat = value(P);
end