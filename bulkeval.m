function F = bulkeval(p,x,X)
%% Description
% evaluate efficiently a polynomial for multiple values
F = [];
B = getbase(p);
[M,vartype] = yalmip('monomtable');

xvar = getvariables(x);
pvar = getvariables(p);
for i = 1:size(X,2)
    xval = X(:,i);
    z = zeros(1,length(vartype));
    z(xvar) = xval;

    m = repmat(z,size(M,1),1).^M;
    m = prod(m,2);

    F = [F B*[1;m(pvar)]];
end
