function [p,x, aux_out] = shape_constrained_regression(algo,degree,features,...
                 response, varargin)
%% Description
% Outputs
%   p := decision polynomial
%   x := argument of p
%   runtime := runtime in seconds for training
% Inputs
%   algo := the regression algoritm - monotone, convex
%   degree := degree of the polynomial
%   features := feature variable data used for training
%   response :=  response variable used for training
%   varargin := optional argument; a sequence of the form:(or any subset of it)
%                      'monotone_profile', ones(3,1),
%                      'convex_sign', -1,
%                      'l_bound', [-1, 2],
%                      'u_bound', [3, 5]

%% Parse the input
[~, k] = size(features);  

%  Define the acceptable names for helper variables in the optional
%  varargin
arg_struct = struct('monotone_profile', ones(k,1),...
                    'convex_sign', 1, ...
                    'l_bound', -ones(k,1),...
                    'u_bound', ones(k,1),...
                    'solver', 'mosek',...
                    'helper_degree', degree-2);
arg_names = fieldnames(arg_struct);

% Count arguments and ensure that they come in pairs
n_args = length(varargin);
if round(n_args/2)~=n_args/2
   error('Property names and property values must come in pairs')
end

% Populate the optional arguments struct with user defined values
for pair = reshape(varargin,2,[]) % pair is {arg; value}
    arg = pair{1}; 

   if any(strcmp(arg, arg_names))
      arg_struct.(arg) = pair{2};
   else
      error('%s is not a recognized parameter name',arg)
   end
end

%% Fit the desired model

switch algo
    case "monotone"
        % fit to the training data
        [p,x, aux_out] = monotone_regression(degree,features,response,...
                                    arg_struct.monotone_profile);
    case "bounded_derivative"
        % fit to the training data
        [p,x, aux_out] = bounded_derivative_regression(degree,features,response,...
                                              arg_struct.l_bound,...
                                              arg_struct.u_bound);                                       
    case "convex"
        % fit to the training data
        [p,x, aux_out] = convex_regression(degree,features,response,...
                                  arg_struct.convex_sign,...
                                  'solver', arg_struct.solver, ....
                                  'helper_degree', arg_struct.helper_degree);   
    case "monotone_convex"
        % fit to the training data
        [p,x, aux_out] = monotone_convex_regression(degree,features,response,...
                                           arg_struct.monotone_profile, ...
                                           arg_struct.convex_sign);   
end
