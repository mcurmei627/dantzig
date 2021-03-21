%% Load image
input_image = imresize(imread('beach.jpg'), 0.3);
[dim1, dim2, ~] = size(input_image);

target_image = imresize(imread('thunder.jpg'),0.001);
input_image = double(reshape(input_image, [],3));
target_image = double(reshape(target_image,[],3));

[num_input_pixels, ~] = size(input_image);
[num_target_pixels, ~] = size(target_image);

[X_mat,~,ic_input] = unique(input_image,'rows');
X_mat = X_mat/256;
a_vec=accumarray(ic_input(:), 1);
a_vec=a_vec/num_input_pixels;

[Y_mat,~,ic_target] = unique(target_image,'rows');
Y_mat = Y_mat/256;
b_vec=accumarray(ic_target(:), 1);
b_vec=b_vec/num_target_pixels;

n = length(a_vec);
m = length(b_vec);
%% Initialization
% a_vec = [0.2, 0.5, 0.3];
% b_vec = [0.5, 0.1, 0.4];
% X_mat = [[1,1]; [1, 0]; [0, 2]];
% Y_mat = [[1, 2]; [0, 1]; [0,1]];
alpha_mat = initialize_gradients(X_mat, Y_mat, a_vec, b_vec);
%% Solve LP for P
% alpha_mat = [[1,2,];[0.5, 1.5]];
% a_vec = [0.5, 0.5];
% b_vec = [0.2, 0.8];
% X_mat = [[1,1, 1]; [0, 0, 2]];
% Y_mat = [[1, 2, 0]; [1, 0,1]];

for i = 1:2
    P = get_optimal_transport_polytope(alpha_mat, a_vec, b_vec);
    %% Solve SDP for f

    degree = 5;
    helper_degree=3;
    small_ell = 1;
    big_ell = 10;
    inf_domain = [0,0,0];
    sup_domain = [1,1,1];

    [f,x, alpha_mat, grad_fx] = get_monge_potential(P, X_mat, Y_mat, degree, helper_degree, ...
        small_ell, big_ell, inf_domain, sup_domain);
    output_image = grad_fx(ic_input,:);
    output_image = max(output_image, 0);
    output_image = min(output_image,1);
    output_image = uint8(round(256*output_image));
    output_image = reshape(output_image, dim1, dim2, 3);
    fig = figure();
    imshow(output_image)
    saveas(fig,'updated','png');
end

