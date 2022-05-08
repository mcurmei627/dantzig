k = 2;
headers = {'N', 'd', 'r', 'min_eig', 'try'};
    output = cell2table(cell(0,5),'VariableNames', headers);
    for t = 1:50
    for N= [100, 1000, 10000]
        features = rand(N, k);
        response_clean = pd_test_fun(features);
        noise = std(response_clean).*normrnd(0,1,[N,1]);
        response_noisy = 0.2*noise + response_clean;
        for d = [4, 6]
            for r = [(d/2-1), d/2]
                [p,x, aux_out] = convex_regression(d,features,response_noisy,1, ...
                    'helper_degree', r);
                min_eig = min(eig(aux_out.('Q')));
                output_row = {N, d, r, min_eig, t};
                output = [output; output_row];
            end
        end
    end
end
        
output.isPD = output.min_eig > 0.0001;
stats = varfun(@mean, output,"InputVariables",'isPD', ...
                            "GroupingVariables",["N", "d", "r"]);  


