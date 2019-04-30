function[R,dR,d2R] = genTikhonov(W,param)
%[R,dR,d2R] = genTikhonov(W,param)
%
%  R(W) = (h/2)* | L*W|_F^2
% 
% where the scalar h adapts for mesh size and L is a matrix (e.g.,
% differential operator)
% 
% Inputs:
%   W      - weights, either as nf x nc matrix or vector (preferred in
%            optimization)
%   param -  struct whose fields control the type of regularizer. 
%            Fields include:
%               nc - number of classes
%               h  - mesh-size, e.g., pixel size in 2D
%               L  - matrix
%               Wref - reference weights

if nargin==0
    testGenTikhonov
    return;
end

vec = @(x) x(:);
h = param.h;
L = param.L;
nc = param.nc;
Wref = param.Wref;
alpha = param.alpha;

W = reshape(W,[],nc);
Wref = reshape(Wref, [], nc);
Wdiff = W-Wref;
R  = ((alpha*h)/2) * norm(L*(Wdiff),'fro')^2;
if nargout>1
    LtL = L'*L;
    dR = alpha*h*vec(LtL*(Wdiff));
end

if nargout > 2
    resh = @(v) reshape(v, [], nc);
    d2R = @(v) vec(alpha*h*(LtL)*resh(v));
end