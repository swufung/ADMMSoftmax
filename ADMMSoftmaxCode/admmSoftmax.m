function [wFinal,wOptLoss,wOptAcc,his] = admmSoftmax(W,param)
% ADMM Softmax function
% inputs: 
%   w0 - initial weights, size(w0) = nf*nc x 1 
%   parameter structure containing input variables

% output: 
%   wOptLoss - weights that lead to smallest loss on validation dataset
%   wOptAcc  - weights that lead to highest accuracy
%   wFinal   - weights at final iteration
%   his      - history matrix. The columns correspond to: 
%              1) iteration number 
%              2) training misfit 
%              3) validation misfit
%              4) training accuracies, 
%              5) validation accuracies, 
%              6) number of newton iterations for z-step 
%              7) z-step misfit 
%              8) relative gradient norm, 
%              9) lagrangian, 
%              10) primal residual, 
%              11) primal tolerance, 
%              12) dual residual, 
%              13) dual tolerance, 
%              14) rho value, 
%              15) current runtime of algorithm


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
% out             = param.out;

Wref            = param.Wref;
Dtrain          = param.Dtrain;
Dval           = param.Dval;
Ctrain          = param.Ctrain;
Cval           = param.Cval;
L               = param.L;

f               = param.f;    % class obj func with training data
fVal            = param.fVal; % class obj func with validation data

% parameters for Z-step solver
atolZ           = param.atolZ;
rtolZ           = param.rtolZ; 
maxIterZ        = param.maxIterZ;
linSolMaxIterZ  = param.linSolMaxIterZ;
linSolTolZ      = param.linSolTolZ;
lsMaxIterZ      = param.lsMaxIterZ;
outZ            = param.outZ;


nc = size(Ctrain,1);
Ntrain = size(Dtrain,2); Nval = size(Dval,2);

DDt = Dtrain*Dtrain';
LLt = L*L';
Z       = W*Dtrain; 

U   = zeros(nc,Ntrain);
rho = rho0; 
rhoOld = rho;
A = rho*DDt + alpha*LLt;
LLtWrefT = LLt*Wref';
iter = 1;
currentRuntime = 0;

his = zeros(maxIter,15);

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

if varRho==1
    fprintf('\t\t\t\t\t ========== adaptive ADMMSoftmax ========== \t\t\t\n')
else
    fprintf('\t\t\t\t\t ========== fixed ADMMSoftmax ========== \t\t\t\n')
end

    fprintf('maxIter=%d, rho=%1.1e, varRho=%d, scaleRho=%1.2f, mu=%1.2f, rtol=%1.2e, atol=%1.2e, alpha=%1.2e, lsSolver = %s \n', ...
        maxIter, rho0, varRho, scaleRho, mu, rtol, atol, alpha, lsSolver);
    fprintf('maxIterZ=%d, linSolMaxIterZ=%d, atolZ=%1.2e, rtolZ=%1.2e, linSolTolZ=%1.2e \n', ...
        maxIterZ, linSolMaxIterZ, atolZ, rtolZ, linSolTolZ);
    fprintf('\niter\tfTrain\t  fVal\t     |W-Wold|\ttrainAcc  valAcc  Ziters    Fz\t      |dJ|/|dJ0|\tLagrangian\tresPri\t   epsPri    resDual\tepsDual\t  rho\t    runtime  iterW  flagW     resW        tLS      tZ\n')

lowestMisfit = Inf;
highestAcc   = 0;
nrm0         = 0;

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
    if iter==1; [fTemp, paraTemp, dF0] =f.pLoss.getMisfit(WD, Ctrain); nrm0 = norm(dF0(:),2); end
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
    Z = reshape(Z, nc, Ntrain);

    tElapsedZ = toc(tStartZ);
    
    Fz = hisZ.his(end,2);



    %% update dual variable
    U = U + (Z - WD);

    %% store values
    WDval = W*Dval;
    [fcTrain, paraTrain, dFTrain] = f.pLoss.getMisfit(WD, Ctrain);
    [fcVal, paraVal]  = fVal.pLoss.getMisfit(WDval, Cval);

    accTrain = 100*(Ntrain-paraTrain(3))/Ntrain; 
    accVal = 100*(Nval-paraVal(3))/Nval;
    
    relGradNorm = norm(dFTrain(:),2)/nrm0;
    
    %% keep weights containing highest accuracy and lowest misfit from validation set
    
    if fcVal<=lowestMisfit
        lowestMisfit = fcVal;
        wOptLoss = W;
    end
    if accVal>=highestAcc
        highestAcc = accVal;
        wOptAcc = W;
    end
    
    %%
    %% primal & dual residual
    resPri  = norm(Z-WD,'fro');
    resDual = norm(rho*Dtrain*(Z - Zold)', 'fro');
    epsPri  = sqrt(numel(Z))*atol + rtol*max(norm(Z,'fro'),norm(WD,'fro'));
    epsDual = sqrt(numel(U))*atol + rtol*norm(U,'fro');

    tElapsedIter = toc(tStartIter);
    currentRuntime = currentRuntime + tElapsedIter;
    
    % lagrangian=Fz+ Reg + 0.5rho*resPri^2 + sum(y_j'(u_j'*(z_j-Wd_j))
    his(iter,9) = Fz + 0.5*alpha*norm(L*(W-Wref)','fro')^2 + 0.5*rho*norm(resPri+U,'fro')^2 - 0.5*rho*norm(U, 'fro')^2;

    Wdiff = norm(W(:) - Wold(:));

    % printing
    fprintf('%d\t%1.2e  %1.2e   %1.2e\t%1.2f\t  %1.2f\t  %d\t%1.2e\t%1.2e\t%1.2e\t%1.2e   %1.2e  %1.2e\t%1.2e  %1.2e  %1.2f      %d       %d     %1.2e  %1.2e     %1.2e\n',...
        iter, fcTrain,fcVal, Wdiff, accTrain, accVal, length(hisZ.his(:,1)), Fz,...
        relGradNorm, his(iter,9), resPri, epsPri, resDual, epsDual, rho, currentRuntime,...
        iterW, flagW, relresW, tElapsedLS, tElapsedZ);
    
    his(iter,1)   = iter; % current iter
    his(iter,2:3) = [fcTrain, fcVal]; % training and validation misfits
    his(iter,4:5) = [accTrain, accVal]; % training and validation accuracies
    
    his(iter,6) = length(hisZ.his(:,1)); % number of newton iters for Z step
    his(iter,7) = Fz; % Fz
%     his(iter,8) = hisZ.his(end,4); % gradient norm at end of Z newton
    his(iter,8) = relGradNorm; % optimality condition
    
    his(iter, 10:15) = [resPri, epsPri, resDual, epsDual, rho, currentRuntime];

    % stopping tolerance
    if strcmp(stoppingCrit{1}, 'regular')
        if resPri<=epsPri && resDual<=epsDual
            his = his(1:iter,:);
            wFinal = W(:);
            break;
        end
    elseif strcmp(stoppingCrit{1}, 'training')
        if accTrain>=stoppingCrit{2}
            his = his(1:iter,:);
            wFinal = W(:);
            break;
        end
    elseif strcmp(stoppingCrit{1}, 'validation')
    if accVal>=stoppingCrit{2}
        his = his(1:iter,:);
        wFinal = W(:);
        break;
    end
    elseif strcmp(stoppingCrit{1}, 'runtime')
        if currentRuntime>=stoppingCrit{2}
            his = his(1:iter,:);
            wFinal = W(:);
            break;
        end
    elseif strcmp(stoppingCrit{1}, 'relGradNorm')
        if  relGradNorm<=stoppingCrit{2}
            his = his(1:iter,:);
            wFinal = W(:);
            break;
        end
    end
    
    if iter>=maxIter
        his = his(1:iter,:);
        wFinal = W(:);
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

