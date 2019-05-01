clear all; clc;

addpath(genpath('~/ADMMSoftmaxCode'))

clear all; 
N = 50000; Nval = 0.2*N;
[Dtrain,Ctrain,Dval,Cval] = setupMNIST(N, Nval);

% Dtrain, Ctrain = training data
% Dval, Cval = validation data

Dtrain = reshape(Dtrain, 28*28, N);
Dval = reshape(Dval, 28*28, Nval); 

fprintf('maxY = %1.2e, minY = %1.2e', max(Dtrain(:)), min(Dtrain(:)));

channelsIn = 1; 
channelsOut = 9;
nImg = [28 28];

fprintf(' number of training examples: %d \n\n', N);


%% extreme learning
% kernel size
sK = [3, 3, channelsIn, channelsOut]; %3x3 convolution window
Ker = convFFT(nImg, sK);
th   = randn(nTheta(Ker),1);
K  = getOp(Ker,th);
Dtrain = tanh(K*Dtrain); Dval = tanh(K*Dval);

Dtrain = reshape(Dtrain, [], N); Dval = reshape(Dval, [], Nval);

nf = size(Dtrain,1); nc = size(Ctrain,1);

%% start optimization

addBias=true;

%% regularization
% smoothness reg. operator
fprintf('using smoothness! reg. operator...\n')
Ltemp = getLaplacian(nImg, 1./nImg);

L = genBlkDiag(Ltemp,channelsOut-1);

%  add bias to laplace operator 
if addBias==true
    L = sparse([L zeros(size(L,1),1); zeros(1,size(L,2)) 1]);
end

Lout = sparse(genBlkDiag(L, nc-1));

% account for mesh size: 
Lout = Lout/(nf);
L    = L/(nf);

fprintf('size of Lout = %d...\n', size(Lout,1))
fprintf('length of W = %d...\n', nf*nc)

alpha = 1e-1; 

%%
% for evaluating misfits accuracies
pRegW   = tikhonovReg(Lout,alpha);
pLoss   = softmaxLoss();
f       = classObjFctn(pLoss,pRegW,Dtrain,Ctrain);
fTest   = classObjFctn(pLoss,pRegW,Dval,Cval);

f.pLoss.addBias=addBias; fTest.pLoss.addBias=addBias;


%% initial admm values
rho0 = 1e-8; 
maxIter = 1000; atol = 1e-12; rtol = 1e-12;
his = zeros(maxIter,14); out=1; varRho=1; scaleRho = 2; mu = 10;
rhoLowerBound = 1e-16;
rhoUpperBound = 1e3;
% 1)iter 2)Fw 3)Ftest, 4)trainAcc 5)testAcc 6)Ziters 7)Fz 8)Zres 9)lagrangian
%10) resPri 11)epsPri 12)resDual 13)epsDual 14)rho 15)currentRuntime

% NOTE: to change from adaptive ADMM to fixed ADMM, switch to varRho=0

%% LeastSquares parameters
% lsSolver = 'cholesky';
lsSolver = 'backslash';
tolW = 1e-2; 
maxIterW = 20; 

%% Z-step parameters
maxIterZ = 100; % max number of Z newton iters
linSolMaxIterZ = 20; % max number of CG iters per newton step in Z step
lsMaxIterZ= 20; % max number of linesearch armijo iters per lin sol in Z step
atolZ = 1e-4; rtolZ=1e-4;
zOut = 0;
linSolTolZ = 1e-4; % tolerance of linear solver (steihaug CG) for Z newton step
%% stopping criteria
% stoppingCrit{1} = 'regular';
% stoppingCrit{1} = 'training'; stoppingCrit{2} = 90; % stop when 90% training
stoppingCrit{1} = 'runtime'; stoppingCrit{2} = 500; % stop after 10 seconds
% stoppingCrit{1} = 'maxiters'; stoppingCrit{2} = 50; % stop after 50 iters



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% initialize 
if addBias==true
    W      = (randn(nc,nf+1));
    Dtrain       = [Dtrain; ones(1,N)];
    Wref    = zeros(nc,nf+1);
else
    W      = (randn(nc,nf));
    Wref    = zeros(nc,nf);
end

DDt = Dtrain*Dtrain';
LLt = L*L';
Z       = W*Dtrain; 

U   = zeros(nc,N);
rho = rho0; 

A = rho*DDt + alpha*LLt;

if strcmp(lsSolver, 'cholesky')
    tic()
    fprintf(' using %s ls solver...\n', lsSolver);
    C = chol(A);
    toc();
end

if out>=1
        fprintf('\t\t\t\t\t ========== ADMMSoftmax ========== \t\t\t\n')
        fprintf('maxIter=%d, rho=%1.1e, varRho=%d, scaleRho=%1.2f, mu=%1.2f, rtol=%1.2e, atol=%1.2e, alpha=%1.2e, lsSolver = %s \n', ...
            maxIter, rho0, varRho, scaleRho, mu, rtol, atol, alpha, lsSolver);
        fprintf('maxIterZ=%d, linSolMaxIterZ=%d, atolZ=%1.2e, rtolZ=%1.2e, linSolTolZ=%1.2e \n', ...
            maxIterZ, linSolMaxIterZ, atolZ, rtolZ, linSolTolZ);
        fprintf('\niter\tfTrain\t  fTest\t     |W-Wold|\ttrainAcc  valAcc  Ziters    Fz\t      Zres\tLagrangian\tresPri\t   epsPri    resDual\tepsDual\t  rho\t    runtime  iterW  flagW     resW        tLS      tZ\n')
end

LLtWrefT = LLt*Wref';


iter = 1;
currentRuntime = 0;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADMM LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iter<=maxIter


    tCurrentIter = tic;

    %% solve LS
    tLS = tic;
    Wold = W;
    rhs = (rho*Dtrain*(Z+U)' + alpha*LLtWrefT);
    if strcmp(lsSolver, 'cholesky')
        W = C\(C'\rhs); 
        flagW = 0; relresW = norm(vec(A*W - rhs))/norm(W(:)); iterW = 0;
        W = W'; % transpose back

    elseif strcmp(lsSolver, 'backslash')
        A = rho*DDt + alpha*LLt;
%         W = (A+1e-3*speye(size(A(:,1),1)))\rhs;
        W = A\rhs;
        flagW = 0; relresW = norm(vec(A*W - rhs))/norm(W(:)); iterW = 0;
        W = W'; % transpose back
    end

    tLS = toc(tLS);

    % get current misfit and accuracies (remove the bias for trainY)
    % note: transpose first to match meganet dimension version
    if addBias==true
        [fTrain, paraTrain] = f.pLoss.getMisfit(vec(W), Dtrain(1:end-1,:), Ctrain);
        [fVal, paraVal]  = fTest.pLoss.getMisfit(vec(W), Dval, Cval);
    else
        [fTrain, paraTrain] = f.pLoss.getMisfit(vec(W), Dtrain, Ctrain);
        [fVal, paraVal]  = fTest.pLoss.getMisfit(vec(W), Dval, Cval);
    end

    accTrain = 100*(N-paraTrain(3))/N; accVal = 100*(Nval-paraVal(3))/Nval;

    his(iter,1)   = iter;
    his(iter,2:3) = [fTrain, fVal]; 
    his(iter,4:5) = [accTrain, accVal];

 %% solve Z step (no validation)

    % create current Z regularizer
    tZ = tic;
    WD = W*Dtrain;
    Zref = WD - U;
    pRegZ   = tikhonovReg(opEye(size(Zref,1)*size(Zref,2)),rho);
    pLossZ   = softmaxLossZ();

    fZ       = classObjFctnZ(pLossZ,pRegZ,Ctrain);
%     Zref = Zref'; Z = Z';
%     fZ = classObjFctn(pLossZ,pRegZ,eye(nc),Ctrain');

    fZ.pLoss.addBias=0; 

    optZ      = newton();
    optZ.atol = atolZ;
    optZ.rtol = rtolZ;
    optZ.stoppingTime = 1e10;
    optZ.maxIter= maxIterZ;
    optZ.LS.maxIter=lsMaxIterZ;
    optZ.linSol.maxIter=linSolMaxIterZ;
    optZ.linSol.tol = linSolTolZ;
    optZ.out = zOut;

    Zold = Z;
%     Zold = Z';
    [Z, hisZ] = solve(optZ,fZ,Z(:));
    Z = reshape(Z, nc, N);
%     Z = reshape(Z, N, nc); Z = Z';

    tZ = toc(tZ);


    his(iter,6) = length(hisZ.his(:,1)); % number of newton iters for Z step
    his(iter,7) = hisZ.his(end,2); % Fz
    his(iter,8) = hisZ.his(end,4); % gradient norm at end of Z newton

    % do not take new Z if Zres>=1 (residual of dJ/dJ0 in optZ
%     Zres = his(iter,8);
%     if Zres>=1; fprintf('\n not taking new Z...\n'); Z = Zold; end


    %% update dual variable
    U = U + (Z - WD);

    %% primal & dual residual
    resPri  = norm(Z-WD,'fro');
    resDual = norm(rho*Dtrain*(Z - Zold)', 'fro');
    epsPri  = sqrt(numel(Z))*atol + rtol*max(norm(Z,'fro'),norm(WD,'fro'));
    epsDual = sqrt(numel(U))*atol + rtol*norm(U,'fro');

    tCurrentIter = toc(tCurrentIter);
    currentRuntime = currentRuntime + tCurrentIter;
    
    % lagrangian=Fz+ Reg + 0.5rho*resPri^2 + sum(y_j'(u_j'*(z_j-Wd_j))
    his(iter,9) = his(iter,7) + 0.5*alpha*norm(L*(W-Wref)','fro')^2 + 0.5*rho*resPri^2 + rho*sum(sum((U.*(Z-WD))));

    his(iter, 10:15) = [resPri, epsPri, resDual, epsDual, rho, currentRuntime];

    Wdiff = norm(W(:) - Wold(:));

    % printing
    if out>=1
        fprintf('%d\t%1.2e  %1.2e   %1.2e\t%1.2f\t  %1.2f\t  %d\t   %1.2e   %1.2e\t%1.2e\t%1.2e   %1.2e  %1.2e\t%1.2e  %1.2e  %1.2f      %d       %d     %1.2e  %1.2e     %1.2e\n',...
            his(iter,1), his(iter,2), his(iter,3), Wdiff, his(iter,4), his(iter,5), his(iter,6), his(iter,7),...
            his(iter,8), his(iter,9), his(iter,10), his(iter,11), his(iter,12), his(iter,13), his(iter,14), his(iter,15),...
            iterW, flagW, relresW, tLS, tZ);
    end

    % stopping tolerance
    if strcmp(stoppingCrit{1}, 'regular')
        if resPri<=epsPri && resDual<=epsDual
            flag=1;
            his = his(1:iter,:);
            break;
        end
    elseif strcmp(stoppingCrit{1}, 'training')
        if stoppingCrit{2}<=trainingAcc
            flag=1; 
            his = his(1:iter,:);
            break;
        end
    elseif strcmp(stoppingCrit{1}, 'runtime')
        if stoppingCrit{2}<=currentRuntime
            flag=1; 
            his = his(1:iter,:);
            break;
        end
    end

    % vary Rho
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

saveResults = 0;
if saveResults==1
    save('admmResultsMNISTAdapt.mat', 'his', 'W', 'alpha', 'atol', 'rtol', 'atolZ', 'rtolZ', 'linSolMaxIterZ', 'lsMaxIterZ', 'maxIterZ', 'rho0', 'tolW')
end
