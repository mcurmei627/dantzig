function mdl = cobb_douglas_fit(features, response)
%% Cobb-Douglass Production function

%%

% Take log transforms of the data
features = log(features);
response = log(response);

% Fit a linear model
mdl = fitlm(features,response','linear');

end