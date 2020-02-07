%% Comparative plots  where we vary the noise level within a plot
%%
comp_plot_by_noise("convex",N,k,3,eps_arr, 5)
% comp_plot_by_noise("convex",N,k,4,eps_arr, 5)
% comp_plot_by_noise("convex",N,k,5,eps_arr, 5)
% comp_plot_by_noise("convex",N,k,6,eps_arr, 5)
comp_plot_by_noise("convex",N,k,7,eps_arr, 5)
%%
comp_plot_by_noise("monotone",N,k,3,eps_arr, 5)
% comp_plot_by_noise("monotone",N,k,4,eps_arr,5)
% comp_plot_by_noise("monotone",N,k,5,eps_arr,5)
comp_plot_by_noise("monotone",N,k,6,eps_arr,5)
% comp_plot_by_noise("monotone",N,k,7,eps_arr,5)
%%
comp_plot_by_noise("monotone_convex",N,k,3,eps_arr, 5)
% comp_plot_by_noise("monotone_convex",N,k,4,eps_arr, 5)
% comp_plot_by_noise("monotone_convex",N,k,5,eps_arr, 5)
comp_plot_by_noise("monotone_convex",N,k,6,eps_arr, 5)
% comp_plot_by_noise("monotone_convex",N,k,7,eps_arr, 5)
%%
comp_plot_by_noise("bounded_derivative",N,k,3,eps_arr, 5, 'l_bound', zeros(k,1))
% comp_plot_by_noise("bounded_derivativex",N,k,4,eps_arr,5, 'l_bound', zeros(k,1))
% comp_plot_by_noise("bounded_derivative",N,k,5,eps_arr, 5, 'l_bound', zeros(k,1))
comp_plot_by_noise("bounded_derivative",N,k,6,eps_arr, 5, 'l_bound', zeros(k,1))
% comp_plot_by_noise("bounded_derivative",N,k,7,eps_arr, 5, 'l_bound', zeros(k,1))
%% Comparative plots where we vary the  degree level within a plot
%%
comp_plot_by_degree("convex",N,k,2,7,0, 5)
comp_plot_by_degree("convex",N,k,2,7,0.1)
comp_plot_by_degree("convex",N,k,2,7,0.3)
comp_plot_by_degree("convex",N,k,2,7,0.7)
comp_plot_by_degree("convex",N,k,2,7,1.2)
