%% Make comparative plots
N_arr = 200:200:400;
eps_arr = 0:1:2;
N_trials = 2;
degree_arr = 2:2:7;
algos = ["monotone", "convex", "monotone_convex", "bounded_derivative"];
k = 3;
N_degrees = length(degree_arr);
N_eps = length(eps_arr);
N_N = length(N_arr);
%% Make comparative plots when varying noise levels within a plot
for i = 1:N_degrees
    degree = degree_arr(i);
    for j = 1:N_N
        N = N_arr(j);
        comp_plot_by_noise("convex",N,k,degree,eps_arr, N_trials)
        comp_plot_by_noise("monotone",N,k,degree,eps_arr, N_trials)
        comp_plot_by_noise("monotone_convex",N,k,degree,eps_arr, N_trials)
        comp_plot_by_noise("bounded_derivative",N,k,degree,eps_arr,...
                            N_trials, 'l_bound', zeros(k,1))
    end
end
%% Comparative plots where we vary the  degree level within a plot
for i = 1:N_eps
    epsilon = eps_arr(i);
    for j = 1:N_N
        N = N_arr(j);
        comp_plot_by_degree("convex",N,k,2,7,epsilon,N_trials)
        comp_plot_by_degree("monotone",N,k,2,7,epsilon,N_trials)
        comp_plot_by_degree("monotone_convex",N,k,2,7,epsilon,N_trials)
        comp_plot_by_degree("bounded_derivative",N,k,2,7,epsilon,...
                             N_trials, 'l_bound', zeros(k,1))
    end
end
%% Projection plots
for i = 1:N_eps
    epsilon = eps_arr(i);
    for j = 1:N_N
        N = N_arr(j);
        comp_projection_plot("convex",N,k,2,7,epsilon,N_trials)
        comp_projection_plot("monotone",N,k,2,7,epsilon,N_trials)
        comp_projection_plot("monotone_convex",N,k,2,7,epsilon,N_trials)
        comp_projection_plot("bounded_derivative",N,k,2,7,epsilon,...
                             N_trials, 'l_bound', zeros(k,1))
    end
end