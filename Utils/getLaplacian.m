function[L] = getLaplacian(n,h)
%[L] = getLaplacian(n,h)
%
if nargin==0
   n = [32 32];
   h = [1 1]./n;
   x = h/2:h:1;
   [X,Y] = ndgrid(x);
   
   u    = cos(pi*X(:).*Y(:));
   Lapu = -pi^2*(X(:).^2+Y(:).^2).*u;
   
   Lap = feval(mfilename,n,h);
   Laput = Lap*u;
    
  figure(1); clf;
  subplot(1,3,1);
  imagesc(reshape(Lapu,n));
  cax = caxis;
  subplot(1,3,2);
  imagesc(reshape(Laput,n));
  caxis(cax);
  subplot(1,3,3);
  imagesc(reshape(Laput-Lapu,n));
  
end

d2dx = 1/h(1)^2*spdiags(ones(n(1),1)*[1  -2  1],-1:1,n(1),n(1));
d2dy = 1/h(2)^2*spdiags(ones(n(2),1)*[1  -2  1],-1:1,n(2),n(2));

L = kron(speye(n(2)),d2dx) + kron(d2dy,speye(n(1))); 
