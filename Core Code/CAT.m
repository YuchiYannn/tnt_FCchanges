function R = CAT(M,Type,Parameter,Parameter_plus)
%% Clustering Analysis Toolkit (CAT)
% 
% - Z.K.X. 2018/06/09
%---------------------------------------------------------------------------------------------%
%% Input
%-- Type
%   [1] Adaptive (/Original) Affinity Propagation Clustering ('AP')
%   [2] k-means Clustering by MATLAB Built-in Function ('k-means')
%-- M
%   [1] Adaptive (/Original) Affinity Propagation Clustering ('AP') 
%        If only s similarity values are known, where s < s^2-s, 
%        they can be passed to 'AP' in an s * 3 matrix M, 
%        where each row of s contains a pair of data point indices 
%        and a corresponding similarity value: M(j,3) is the
%        similarity of data point M(j,1) to data point M(j,2).
%   [2] k-means Clustering by MATLAB Built-in Function ('k-means')
%        Data, specified as a numeric matrix. 
%        The rows of X correspond to observations, and the columns correspond to variables.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%-- Parameter:commonly used parameters
%   [1] Adaptive (/Original) Affinity Propagation Clustering ('AP') 
%       Parameter.algorithm: algorithm type
%                 [1 - adaptive AP (default); 0 - original AP]  
%       Parameter.cut: after clustering, drop an cluster with number of samples < cut
%                 [default value = 5]
%       Parameter.splot: observing a clustering process when it is on
%                 ['plot' - yes (default); 'noplot' - no]
%       Parameter.nrun: max iteration times
%                 [default value of adaptive AP = 50000; default value of original AP = 2000]
%   [2] k-means Clustering by MATLAB Built-in Function ('k-means')
%       Parameter.k: number of clusters
%                 [default value is 2]
%       Parameter.dis: distance metric
%                 ['sqeuclidean' - squared Euclidean distance (default);
%                  'cityblock' - sum of absolute differences; 
%                  'cosine' - one minus the cosine of the included angle between points
%                  'correlation' - one minus the sample correlation between points;
%                  'hamming' - only suitable for binary data]
%       Parameter.rep: number of times to repeat clustering using new initial cluster centroid positions
%                 [default value is 1]
%       Parameter.fig: if generate a Silhouette plot
%                  [1 - yes (default); 0 - no]
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%-- Parameter_plus:least frequently used parameters
%  [1] Adaptive (/Original) Affinity Propagation Clustering ('AP') 
%       Parameter_plus.nrow: the number of data points
%       Parameter_plus.p: preference that a given point can be chosen as a cluster center
%                 [default value is similarity median of similarity]
%       Parameter_plus.nrun: max iteration times
%                 [default value of adaptive AP = 50000; default value of original AP = 2000]
%       Parameter_plus.nconv:convergence condition         
%                 [default value = 50, bigger value means more srict]
%       Parameter_plus.pstep: decreasing step of parameter "preferences" - pstep*pmedian
%                 [default value = 0.01, the searching of "preferences" is finer and the 
%                  algorithm runs slower if pstep is smaller]
%       Parameter_plus.lam: initial damping factor
%                 [default value = 0.5]
%  [2] k-means Clustering by MATLAB Built-in Function ('k-means')
%       Parameter_plus.maxiter: maximum number of iterations
%                 [default value is 1]
%       Parameter_plus.start: method for choosing initial cluster centroid positions
%                 ['cluster'; 'plus' (default); 'sample'; 'uniform'; numeric matrix; numeric array]
%       Parameter_plus.metric: plots the silhouettes using the inter-point distance function 
%                 ['Euclidean'; 'sqEuclidean' (default); 'cityblock'; 'cosine'; 'correlation';
%                 'Hamming'; 'Jaccard']
%       ##########
%       < Note: for more details, see MATLAB documentation of "kmeans" and "silhouette"
%               https://ww2.mathworks.cn/help/stats/kmeans.html#buefs04-Start 
%               https://ww2.mathworks.cn/help/stats/silhouette.html#f3899431 >
%---------------------------------------------------------------------------------------------%
%% Output
%  [1] Adaptive (/Original) Affinity Propagation Clustering
%
%      OCL: the optimal clusting labels
%      OCL_Silhouette: the corresponding Silhouette index of OCL
%      OCL_Silhouette: the corresponding minimal value of Silhouette index of OCL
%
%      [NC] [NCs] [labels] [NCopt] [Silhouette] [Silhouette_min]
%
%      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%      The class labels (clustering solutions) at every number of clusters "NC" 
%      are stored in variable "labels" corresponding to every "NC" in 
%      variable "NCs". The optimal "NC" is found and stored in variable 
%      "NCopt", and the optimal clustering solution is the class labels in 
%      variable "labels" corresponding to "NCopt".
%      The Silhouette indices corresponding to every "NC" in variable "NCs" 
%      are stored in variable "Silhouette", and in "Silmin_min" for the minimal value 
%      of Silhouette indices of any pair of clusters.
%  [2] k-means Clustering by MATLAB Built-in Function ('k-means')
%      OCL: cluster indices
%      Silhouette: Silhouette values in the n-by-1 vector
%      OCL_Silhouette: mean of Silhouette
%      centroid_locations: cluster centroid locations
%      sumd: within-cluster sums of point-to-centroid distances
%      distances: distances from each point to every centroid
%---------------------------------------------------------------------------------------------%
%% Reference
%  [1] Adaptive (/Original) Affinity Propagation Clustering 
%      Frey, B. J., & Dueck, D. (2007). 
%      Clustering by passing messages between data points. science, 315(5814), 972-976.
%      Wang, K., Zhang, J., Li, D., Zhang, X., & Guo, T. (2008). 
%      Adaptive affinity propagation clustering. arXiv preprint arXiv:0805.1096.
%---------------------------------------------------------------------------------------------%
%% Dependency
%  [1] Adaptive (/Original) Affinity Propagation Clustering 
%      Adaptive Affinity Propagation clustering (by Kaijun Wang)
%      [https://ww2.mathworks.cn/matlabcentral/fileexchange/
%       18244-adaptive-affinity-propagation-clustering]
%---------------------------------------------------------------------------------------------%
%% Basic Setting
if (nargin < 2) || isempty(Type) == 1
    Type = 'AP';
end
%
if strcmp(Type,'AP')
    if (nargin < 3) || isfield(Parameter,'algorithm') == 0
        Parameter.algorithm = 1;
    end
    if (nargin < 3) || isfield(Parameter,'nrun') == 0
        if Parameter.algorithm == 1
            Parameter.nrun = 50000;
        else
            Parameter.nrun = 2000;
        end
    end
    if (nargin < 3) || isfield(Parameter,'cut') == 0
        Parameter.cut = 5;
    end
    if (nargin < 3) || isfield(Parameter,'splot') == 0
        Parameter.splot = 'no';
    end    
    if (nargin < 4) || isfield(Parameter_plus,'nconv') == 0
        Parameter_plus.nconv = 50;
    end    
    if (nargin < 4) || isfield(Parameter_plus,'pstep') == 0
        Parameter_plus.pstep = 0.01;
    end
    if (nargin < 4) || isfield(Parameter_plus,'lam') == 0
        Parameter_plus.lam = 0.5;
    end   
end
%
if strcmp(Type,'k-means')
    if (nargin < 3) || isfield(Parameter,'k') == 0
        Parameter.k = 2;
    end
    if (nargin < 3) || isfield(Parameter,'dis') == 0
        Parameter.dis = 'sqeuclidean';
    end
    if (nargin < 3) || isfield(Parameter,'fig') == 0
        Parameter.fig = 1;
    end    
    if (nargin < 3) || isfield(Parameter,'rep') == 0
        Parameter.rep = 1;
    end    
    if (nargin < 4) || isfield(Parameter_plus,'maxiter') == 0
        Parameter_plus.maxiter = 1;
    end    
    if (nargin < 4) || isfield(Parameter_plus,'start') == 0
        Parameter_plus.start = 'plus';
    end
    if (nargin < 4) || isfield(Parameter_plus,'metric') == 0
        Parameter_plus.metric = 'sqEuclidean';
    end    
end
%% AP
if strcmp(Type,'AP')
    id = 100; algorithm = Parameter.algorithm; simatrix = 1;
    if Parameter.algorithm == 1
        [labels,NCs,labelid,iend,Sp,Slam,NCfixs] = adapt_apcluster(M,[],...
        [],Parameter_plus.pstep,1,'convits',Parameter_plus.nconv,'maxits',Parameter.nrun,...
        'dampfact',Parameter_plus.lam,Parameter.splot);
        [NC,Sil,Silmin] = solution_evaluation([],M,labels,NCs,...
        NCfixs,1,[],[],Parameter.cut);
        fprintf('## Running iterations = %g \n', iend);
        solution_findK;
        if exist('R')
            clear R;
        end
        R.NC = NC; R.NCs = NCs; R.NCopt = NCopt; R.labels = labels;
        R.Silhouette = Sil; R.Silhouette_min = Silmin;
        F = find(NCs == NCopt);
        if F > length(NC)
            F = length(NC);
        end
        R.OCL = labels(:,F);
        R.OCL_Silhouette = Sil(F);
        R.OCL_Silhouette_min = Silmin(F);
    else
        cut = Parameter.cut;
        if isfield(Parameter_plus,'p') == 0
            dn = find(M(:,3)>-realmax);
            Parameter_plus.p = median(M(dn,3));        
        end
        [labels,netsim,iend,unconverged] = apcluster(M,Parameter_plus.p,'convits',...
        Parameter_plus.nconv,'maxits',Parameter.nrun,'dampfact',...
        Parameter_plus.lam,Parameter.splot);
        if isfield(Parameter,'nrun') ~= 0
            nrow = Parameter_plus.nrow;
        else
            y = size(M,1);
            syms x
            eqn = x^2-x == 2*y;
            solx = solve(eqn,x);
            K = double(solx);
            nrow = K(K>0);
        end
        fprintf('## Running iterations = %g \n', iend);
        solution_findK;
        R.OCL = labels; R.OCL_Silhouette; R.OCL_Silhouette_min = Silmin;
    end
    close;
end
%% k-means (built-in function)
if strcmp(Type,'k-means')
    [idx,C,sumd,D] = kmeans(M,Parameter.k,'Distance',...
        Parameter.dis,'MaxIter',Parameter_plus.maxiter,'Replicates',Parameter.rep,...
        'Start',Parameter_plus.start); 
    if Parameter.fig == 1
        [s,h] = silhouette(M,idx,Parameter_plus.metric);
    else
        s = silhouette(M,idx,Parameter_plus.metric);
    end
    R.OCL = idx; R.OCL_Silhouette = mean(s); R.Silhouette = s;
    R.centroid_locations = C; R.sumd = sumd; R.distances = D;
end
