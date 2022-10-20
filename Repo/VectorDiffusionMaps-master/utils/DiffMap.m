function [rslt] = DiffMap(dm)
%
%
% INPUT:
%   dm.data 	   = pxn matrix, n points in R^p
%   dm.epsilon     = kernel parameter
%   dm.NN    	   = the number of nearest neighbors chosen in the algorithm
%   dm.T     	   = diffusion time
%   dm.delta 	   = truncation threshold
%   dm.symmetrize  = symmetrize the graph
% (OPTIONS)
%   dm.compact       = compact support kernel
%   dm.debug         = debug message
%   dm.embedmaxdim   = largest allowable dimension for embedding
%                    (embedding is mandatory, according to truncation; this 
%                     option is the upper bound for allowable embeddings)
%   dm.cleanup       = clean up before executing the program
%   dm.display       = display results
%
% OUTPUT:
%   rslt.Delta	  = Laplace-Beltrami operator
%   rslt.UDelta	  = eigenfunctions of Laplace-Beltrami
%   rslt.lambdaDelta = eigenvalues of Laplace-Beltrami
%   rslt.embeddim = actual embedded dimension
%   rslt.embedded = embedded data points
%
% DEPENDENCY:
%   none
%
% originally written by Hau-tieng Wu 2011-06-20 (hauwu@math.princeton.edu)
% last modified by Tingran Gao (trgao10@math.duke.edu) 2014-06-24
% 

dm.symmetrize    = getoptions(dm, 'symmetrize', 0);
dm.compact       = getoptions(dm, 'compact', 0);
dm.debug         = getoptions(dm, 'debug', 0);
dm.embedmaxdim   = getoptions(dm, 'embedmaxdim', 100);
dm.cleanup       = getoptions(dm, 'cleanup', 0);
dm.display       = getoptions(dm, 'display', 1);

if dm.debug==1
    fprintf('\n(DEBUG:DM)\t\tStart to work on Diffusion Map\n');
end
if dm.cleanup==1
    clc;
    close all;
end

eigopt.isreal = 1;
eigopt.issym = 1;
eigopt.maxit = 3000;
eigopt.disp = 0;

[pp,nn] = size(dm.data);

%%% nearest neighbor search
if dm.NN<pp+1;
    error('*** ERROR: choose NN>p');
end

if ~isfield(dm,'distance') || ~isfield(dm,'index') || ~isfield(dm,'patchno')
    
    atria = nn_prepare(dm.data.');
    [index,distance] = nn_search(dm.data.',atria,(1:nn).',dm.NN,-1,0.0);
    
    if dm.debug
        fprintf(['(DEBUG:DM) NN = ' num2str(dm.NN),'\n']);
        fprintf(['(DEBUG:DM) minimal farthest distance = ' num2str(min(distance(:,end))),'\n']);
        fprintf(['(DEBUG:DM) maximal farthest distance = ' num2str(max(distance(:,end))),'\n']);
        fprintf(['(DEBUG:DM) median farthest distance = ' num2str(median(distance(:,end))),'\n']);
        fprintf(['(DEBUG:DM) 1.5*sqrt(min farthest distance) = ' num2str(1.5*sqrt(min(distance(:,end)))),'.\n']);
        fprintf(['(DEBUG:DM) 1.5*sqrt(max farthest distance) = ' num2str(1.5*sqrt(max(distance(:,end)))),'.\n']);
        fprintf(['(DEBUG:DM) 1.5*sqrt(median farthest distance) = ' num2str(1.5*sqrt(median(distance(:,end)))),'.\n']);
    end
    
    %%% patchno is set to convert the NN info to the \sqrt{h} info
    patchno = dm.NN*ones(1,nn);
    
    if dm.debug==1
        fprintf('(DEBUG:DM) the neighbors with kernel value less than exp(-5*1.5^2)=1.3e-5 are trimmed.\n');
    end
    
    for ii=1:nn
        patchno(ii) = sum(distance(ii,:) <= 1.5*sqrt(5*dm.epsilon));
        distance(ii, distance(ii,:) > 1.5*sqrt(5*dm.epsilon)) = Inf;
    end
    
    NN = max(patchno);
    distance = distance(:,1:NN);
    index = index(:,1:NN);
    rslt.patchno = patchno;
    rslt.distance = distance;
    rslt.index = index;
    rslt.NN = NN;
    
    if dm.debug==1
        if quantile(patchno,0.9) == dm.NN
            %%% it means the NN is not big enough so that the decay of the
            %%% kernel won't be fast enough for the error to be small
            fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++\n');
            fprintf('(DEBUG:DM:WARNING) the NN should be chosen larger\n');
            fprintf('+++++++++++++++++++++++++++++++++++++++++++++++++\n');
        end
        
        fprintf(['(DEBUG:DM) the number of points with distance less then\n']);
        fprintf(['           1.5*sqrt(5*epsilon)=',num2str(1.5*sqrt(5*dm.epsilon)),...
                 ' is (min,max,median) = (',num2str(min(patchno)),',',...
                 num2str(max(patchno)),',',num2str(median(patchno)),')\n']);
        fprintf(['(DEBUG:DM) set NN to be ',num2str(max(patchno)),'\n']);
        
    end
else
    index = dm.index; distance = dm.distance; patchno = dm.patchno;
    rslt.patchno = patchno; rslt.distance = distance; rslt.index = index;
end

Q = exp(-(distance.^2./dm.epsilon));
Q = Q.';
I = repmat(1:nn, NN, 1);
J = index.';

W   = sparse(I(:), J(:), Q(:), nn, nn, nn*NN);
if dm.symmetrize==1 % not quite necessary, since W.*W' below
    W = min(W,W'); % if W(i,j) is 0, set W(j,i) to be 0 as well
end
W   = W*W';	%% make it symmetric (and positively definite).
                % So diffusion time is doubled!
                
D   = sum(W); D = D(:);
D1  = sparse(1:length(D),1:length(D),1./D); % normalized, alpha=1
Lap = D1*W - eye(size(W));
Lap(abs(Lap)<1e-7)=0;
expLap = expm(-Lap);
expLap(abs(expLap)<1e-7)=0;
W1  = D1*W*D1;	%% alpha = 1
D   = sqrt(sum(W1)); D = D(:);
D2  = sparse(1:length(D),1:length(D),1./D);
W2  = D2*W1*D2;	%% almost symmetric. 1e-18 error
W2  = (W2+W2')/2;  %% artifically added.

if dm.debug==1
    tic;
    fprintf('(DEBUG:DM) Calculate eigenfunctions/eigenvalues...');
end

[UD, lambdaUD] = eigs(W2, dm.embedmaxdim, 'LM', eigopt);

if dm.debug==1
    fprintf('DONE! \n');
    toc;
end

UD = D2*UD; lambdaUD = diag(lambdaUD);
[lambdaUD, sortidx] = sort(lambdaUD,'descend');
UD = UD(:, sortidx);

rslt.UL = UD;
rslt.lambdaL = lambdaUD;
rslt.Laplacian = Lap;
rslt.expLaplacian = expLap;
dimidxs   = find((lambdaUD./lambdaUD(2)).^dm.T > dm.delta);
embeddim = length(dimidxs);
if dm.debug
    fprintf(['(DEBUG:DM) Diffusion map will embed the data to ' ...
        num2str(embeddim) '-dim Euclidean space\n']);
end
rslt.embeddim = embeddim;

embedded = UD(:,2:embeddim)*diag(lambdaUD(2:embeddim).^dm.T);
embedded = embedded';
rslt.embedded = embedded;

if dm.display==1
    figure; bar(1-lambdaUD);
    axis tight; title('eigenvalues up to embedmaxdim'); cameratoolbar;
end



