
headers = {'features', 'N', 'd', 'sos_obj 2r=2', 'sos_obj 2r=4', 'sos_obj 2r=6' 'sample_obj'};
output = cell2table(cell(0,length(headers)),'VariableNames', headers);

for k = [2,3,4]
    features_z = rand(1000, k);
    for N= [100, 1000, 10000]
        features = rand(N, k);
        response_clean = convex_fun(features);
        noise = std(response_clean).*normrnd(0,1,[N,1]);
        response_noisy = 0.2*noise + response_clean;
        for d = [2, 4, 6]
            [~,~, aux_out] = convex_regression(d,features,response_noisy,1, ...
                    'helper_degree', 2);
            sos_obj2 = aux_out.('train_rmse');
            
            [~,~, aux_out] = convex_regression(d,features,response_noisy,1, ...
                    'helper_degree', 4);
            sos_obj4 = aux_out.('train_rmse');
            
            [~,~, aux_out] = convex_regression(d,features,response_noisy,1, ...
                    'helper_degree', 6);
            sos_obj6 = aux_out.('train_rmse');
            
            [~,~, aux_out] = sample_convex_regression(d,features,response_noisy, features_z);
            sample_obj = aux_out.('train_rmse');
            
            output_row = {k, N, d, sos_obj2, sos_obj4, sos_obj6, sample_obj};
            output = [output; output_row];
        end
    end
end 


