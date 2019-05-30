function [wOpt,his] = admmSoftmax(W,param)
% ADMM Softmax function
% inputs: 
%   w0 - initial weights, size(w0) = nf*nc x 1 
%   parameter structure containing input variables


maxIter         = param.maxIter;
stoppingCrit    = param.stoppingCrit;
varRho          = param.varRho;
rhoLowerBound   = param.rhoLowerBound;
rhoUpperBound   = param.rhoUpperBound;
mu              = param.mu;
atol            = param.atol;
rtol            = param.rtol;
alpha           = param.alpha;
lsSolver        = param.lsSolver;
rho0            = param.rho0;
scaleRho        = param.scaleRho;
out             = param.out;

Wref            = param.Wref;
Dtrain          = param.Dtrain;
Dtest           = param.Dtest;
Ctrain          = param.Ctrain;
Ctest           = param.Ctest;
L               = param.L;

f               = param.f;    % class obj func with training data
fTest           = param.fTest; % class obj func with test data

% parameters for Z-step solver
atolZ           = param.atolZ;
rtolZ           = param.rtolZ; 
maxIterZ        = param.maxIterZ;
linSolMaxIterZ  = param.linSolMaxIterZ;
linSolTolZ      = param.linSolTolZ;
lsMaxIterZ      = param.lsMaxIterZ;
outZ            = param.outZ;


nc = size(Ctrain,1);
N = size(Dtrain,2); Ntest = size(Dtest,2);

DDt = Dtrain*Dtrain';
LLt = L*L';
Z       = W*Dtrain; 

U   = zeros(nc,N);
rho = rho0; 
rhoOld = rho;
A = rho*DDt + alpha*LLt;
LLtWrefT = LLt*Wref';
iter = 1;
currentRuntime = 0;

his = zeros(maxIter,14);

if strcmp(lsSolver, 'cholesky')
    tStartChol = tic;
    fprintf(' using %s ls solver...\n', lsSolver);
    C = chol(A);
    tElapsedChol = toc(tStartChol)
elseif strcmp(lsSolver, 'qr')
    tStartQR = tic;
     fprintf(' using %s ls solver...\n', lsSolver);
    [Q,R] = qr(A);
    tElapsedQR = toc(tStartQR)
end

if out>=1
    if varRho==1
        fprintf('\t\t\t\t\t ========== adaptive ADMMSoftmax ========== \t\t\t\n')
    else
        fprintf('\t\t\t\t\t ========== fixed ADMMSoftmax ========== \t\t\t\n')
    end
    
        fprintf('maxIter=%d, rho=%1.1e, varRho=%d, scaleRho=%1.2f, mu=%1.2f, rtol=%1.2e, atol=%1.2e, alpha=%1.2e, lsSolver = %s \n', ...
            maxIter, rho0, varRho, scaleRho, mu, rtol, atol, alpha, lsSolver);
        fprintf('maxIterZ=%d, linSolMaxIterZ=%d, atolZ=%1.2e, rtolZ=%1.2e, linSolTolZ=%1.2e \n', ...
            maxIterZ, linSolMaxIterZ, atolZ, rtolZ, linSolTolZ);
        fprintf('\niter\tfTrain\t  fTest\t     |W-Wold|\ttrainAcc  testAcc  Ziters    Fz\t      Zres\tLagrangian\tresPri\t   epsPri    resDual\tepsDual\t  rho\t    runtime  iterW  flagW     resW        tLS      tZ\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADMM LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iter<=maxIter

    tStartIter = tic;
    
    %% solve LS
    tStartLS = tic;
    Wold = W;
    rhs = (rho*Dtrain*(Z+U)' + alpha*LLtWrefT);
    if strcmp(lsSolver, 'cholesky')
        % if rho did not change, do not re-factorize
        if rhoOld~=rho
            A = rho*DDt + alpha*LLt;
            C = chol(A);
        end
        W = C\(C'\rhs); 
        flagW = 0; relresW = norm(vec(A*W - rhs))/norm(W(:)); iterW = 0;
        W = W'; % transpose back
        
    elseif strcmp(lsSolver, 'qr')
        % if rho did not change, do not re-factorize
        if rhoOld~=rho
            A = rho*DDt + alpha*LLt;
            [Q,R] = qr(A);
        end
        W = R\(Q'*rhs);
        flagW = 0; relresW = norm(vec(A*W - rhs))/norm(W(:)); iterW = 0;
        W = W';
    elseif strcmp(lsSolver, 'backslash')
        A = rho*DDt + alpha*LLt;
        W = A\rhs;
        flagW = 0; relresW = norm(vec(A*W - rhs))/norm(W(:)); iterW = 0;
        W = W';
    end

    tElapsedLS = toc(tStartLS);
    
    %% solve Z step (no validation)

    % create current Z regularizer
    tStartZ = tic;
    WD      = W*Dtrain;
    Zref = WD - U;
    pRegZ   = tikhonovReg(opEye(size(Zref,1)*size(Zref,2)),rho, Zref(:));
    pLossZ  = softmaxLossZ();

    fZ       = classObjFctnZ(pLossZ,pRegZ,Ctrain);

    fZ.pLoss.addBias=0; 

    optZ      = newton();
    optZ.atol = atolZ;
    optZ.rtol = rtolZ;
    optZ.maxIter= maxIterZ;
    optZ.LS.maxIter=lsMaxIterZ;
    optZ.linSol.maxIter=linSolMaxIterZ;
    optZ.linSol.tol = linSolTolZ;
    optZ.out = outZ;

    Zold = Z;
    [Z, hisZ] = solve(optZ,fZ,Z(:));
    Z = reshape(Z, nc, N);

    tElapsedZ = toc(tStartZ);



    %% update dual variable
    U = U + (Z - WD);

    %% store values
    WDtest = W*Dtest;
    [fTrain, paraTrain] = f.pLoss.getMisfit(WD, Ctrain);
    [fcTest, paraVal]  = fTest.pLoss.getMisfit(WDtest, Ctest);

    accTrain = 100*(N-paraTrain(3))/N; accVal = 100*(Ntest-paraVal(3))/Ntest;

    his(iter,1)   = iter; % current iter
    his(iter,2:3) = [fTrain, fcTest]; % training and testing misfits
    his(iter,4:5) = [accTrain, accVal]; % training and testing accuracies
    
    his(iter,6) = length(hisZ.his(:,1)); % number of newton iters for Z step
    his(iter,7) = hisZ.his(end,2); % Fz
    his(iter,8) = hisZ.his(end,4); % gradient norm at end of Z newton
    
    
    %% primal & dual residual
    resPri  = norm(Z-WD,'fro');
    resDual = norm(rho*Dtrain*(Z - Zold)', 'fro');
    epsPri  = sqrt(numel(Z))*atol + rtol*max(norm(Z,'fro'),norm(WD,'fro'));
    epsDual = sqrt(numel(U))*atol + rtol*norm(U,'fro');

    tElapsedIter = toc(tStartIter);
    currentRuntime = currentRuntime + tElapsedIter;
    
    % lagrangian=Fz+ Reg + 0.5rho*resPri^2 + sum(y_j'(u_j'*(z_j-Wd_j))
    his(iter,9) = his(iter,7) + 0.5*alpha*norm(L*(W-Wref)','fro')^2 + 0.5*rho*resPri^2 + rho*sum(sum((U.*(Z-WD))));
    
    his(iter, 10:15) = [resPri, epsPri, resDual, epsDual, rho, currentRuntime];

    Wdiff = norm(W(:) - Wold(:));

    % printing
    if out>=1
        fprintf('%d\t%1.2e  %1.2e   %1.2e\t%1.2f\t  %1.2f\t  %d\t   %1.2e   %1.2e\t%1.2e\t%1.2e   %1.2e  %1.2e\t%1.2e  %1.2e  %1.2f      %d       %d     %1.2e  %1.2e     %1.2e\n',...
            his(iter,1), his(iter,2), his(iter,3), Wdiff, his(iter,4), his(iter,5), his(iter,6), his(iter,7),...
            his(iter,8), his(iter,9), his(iter,10), his(iter,11), his(iter,12), his(iter,13), his(iter,14), his(iter,15),...
            iterW, flagW, relresW, tElapsedLS, tElapsedZ);
    end

    % stopping tolerance
    if strcmp(stoppingCrit{1}, 'regular')
        if resPri<=epsPri && resDual<=epsDual
            his = his(1:iter,:);
            wOpt = W;
            break;
        end
    elseif strcmp(stoppingCrit{1}, 'training')
        if stoppingCrit{2}<=trainingAcc
            his = his(1:iter,:);
            wOpt = W;
            break;
        end
    elseif strcmp(stoppingCrit{1}, 'runtime')
        if stoppingCrit{2}<=currentRuntime
            his = his(1:iter,:);
            wOpt = W;
            break;
        end
    end
    
    if iter>=maxIter
        his = his(1:iter,:);
            wOpt = W;
            break;
    end

    % vary Rho
    rhoOld = rho;
    if varRho==1 && resDual~=0
        if resPri>mu*resDual && resDual~=0
            rho =rho*scaleRho;
            U   = U/scaleRho;
        elseif resDual>mu*resPri && resPri~=0
            rho = rho/scaleRho;
            U   = U*scaleRho;
        end
    end

    % check lower bound
    if rho<rhoLowerBound
        rho=rhoLowerBound;
        U = U*rho/rhoLowerBound;
    elseif rho>rhoUpperBound
        rho=rhoUpperBound;
        U = U*rho/rhoUpperBound;
    end

   iter = iter + 1; 
end
end

