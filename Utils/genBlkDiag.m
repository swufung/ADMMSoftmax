function Mblk = genBlkDiag(M, n)
% given a matrix M and a integer n, creates block diagonal matrix
% blkDiag(M,M,...,M) n times.
    
    Mblk = M;
    for j=1:n
        Mblk = addBlock(Mblk, M);
    end
    
end




function MM =  addBlock(M1, M2)
% adds one block diagonal

    MM = blkdiag(M1,M2);

end