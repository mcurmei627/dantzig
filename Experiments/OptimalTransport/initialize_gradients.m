function alpha_mat = initialize_gradients(X_mat, Y_mat, a_vec, b_vec)
P = initialize_P(a_vec, b_vec);
[n, k] = size(X_mat); 
[m, ~] = size(Y_mat);
grad_fx = [];
for i = 1:n
    curr_grad = sum(1/a_vec(i)*repmat(P(i,:), k, 1).*Y_mat', 2);
    grad_fx = [grad_fx, curr_grad];
end
grad_eval = repmat(grad_fx', m, 1);
y_vals = repelem(Y_mat', 1, n)';
diff = grad_eval'-y_vals';
alpha_vec = vecnorm(diff);
alpha_mat = reshape(alpha_vec, n, m);
end
