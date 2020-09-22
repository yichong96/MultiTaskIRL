% Calculate prior of cluster assignment c, pr(c|alpha)
%
function pr = calClustAssignProb(belongTo, alpha)
format long

% z(i) = # of clusters that are size i
%P(c|alpha) = alpha^K / alpha^(N) Prod (|C| - 1)!)

z  = ownCount(belongTo);
ix = 1:length(z);
Z  = sum(z.*ix);

k  = vpa(ix) .^ vpa(z) .*factorial(vpa(z));
k
pr = factorial(vpa(Z))/prod(vpa(alpha:alpha+Z-1))*(alpha^sum(vpa(z)))/prod(vpa(k));
pr

% my own understanding of how to calculate p(c|alpha)
%==============================================================
%k = factorial(ix - 1) .^ z;
%pr = prod(alpha^sum(z)) / prod(alpha:alpha + Z - 1) * prod(k);


end


%N = alpha * alpha + 1 * alpha + 2 * ... * alpha + N
%prod(alpha:alpha+Z-1) * alpha^sum(z).
%sum(z) = no. of