function [pcaBASIS,estdim] = LocalPCA(lpca)
%
% dimension estimation by local PCA ver 0.3
%
% INPUT:
%   lpca.data         = pxn matrix, n points in R^p
%   lpca.epsilonpca   = kernel parameter (epsilonpca_PCA)
%   lpca.NN        = the number of nearest neighbors chosen in the PCA step
%		             should be large enough.
%   lpca.index	   = output index matrix from nn_search
%   lpca.distance  = output distance matrix from nn_search
% (OPTION)
%   lpca.KN	   = using KN algorithm or not. If =1, then use KN, If<1
%		     then lpca.KN is used as the threshold_gamma
%   lpca.debug     = debug mode.
%
% OUTPUT:
%   pcaBASIS	= eigenfunctions of Laplace-Beltrami
%   estdim	= estimated dimension
%
% DEPENDENCY:
%   none
%
% originally written by Hau-tieng Wu 2011-06-20 (hauwu@math.princeton.edu)
% last modified by Tingran Gao (trgao10@math.duke.edu) 2014-06-24
%

lpca.KN    = getoptions(lpca, 'KN', 0);
lpca.debug = getoptions(lpca, 'debug', 1);

if (lpca.debug == 1)
    fprintf(['(DEBUG:lpca) NN=',num2str(lpca.NN),'; kernel parameter=',num2str(lpca.epsilonpca),'\n']);
end

[pp,nn] = size(lpca.data);

if lpca.KN<1 && lpca.KN>0
    threshold_gamma = lpca.KN;
    if lpca.debug==1
        fprintf('(DEBUG:lpca) Use threshold_gamma to determine the local dimension\n');
    end
else %% lpca.KN>1 || lpca.KN<0
    error('threshold_gamma should be inside [0,1]');
end

if (lpca.debug == 1)
    if pp>nn
        fprintf('(DEBUG:lpca) It is better to do dimension reduction, for example, PCA, to reduce the ambient space dimention\n');
    end
end

pcaBASIS = zeros(pp,pp,nn);
ldim = ones(nn,1);

data	= lpca.data;
distance= lpca.distance;
index	= lpca.index;
NN	= lpca.NN;
patchno = lpca.patchno;

%%% spectral PCA (lpca.KN<1)
Q = exp(-5*(distance.^2./lpca.epsilonpca));
Q = Q.';
I = repmat(1:nn, NN, 1);
J = index.';

W = sparse(I(:), J(:), Q(:), nn, nn, nn*NN);
D = sum(W); D = D(:);

for ii=1:nn
    CENTER = data(:,ii);
    Aii = data(:, index(ii, 2:patchno(ii)))...
        -repmat(CENTER, 1, patchno(ii)-1);
    Dii = diag(W(ii, index(ii, 2:patchno(ii))));
    Cor = Aii*Dii./sqrt(D(ii));
    [U, lambda, ~] = svd(Cor);
    
    if (size(lambda,1)~=1)
        lambda = diag(lambda);
    end
    totalenergy = sum(lambda);
    ldim(ii) = 1;
    energy = 0;
    
    while true
        energy = energy+lambda(ldim(ii));
        if (energy/totalenergy > threshold_gamma)
            break;
        end
        ldim(ii) = ldim(ii)+1;
    end
    pcaBASIS(:,:,ii) = U(:,:);
end

estdim = median(ldim);
if lpca.debug
    fprintf(['(DEBUG:lpca)\t estimated dimension = ',num2str(estdim),'\n']);
end

pcaBASIS = pcaBASIS(:,1:estdim,:);
if (lpca.debug == 1)
    figure;
    hist(estdim, 100); axis tight; axis([0 10 -inf inf])
    set(gca,'fontsize',10);
    title('(DEBUG:lpca) the histogram of the estimated dimension at each point');
end

