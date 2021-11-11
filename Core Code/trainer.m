function R = trainer(X,Y,C,S_type,M_type,Parameter)
%% Model Training Module
%------------------------------------------------------------------------------------------------%
% - Z.K.X. 2018/05/23
%------------------------------------------------------------------------------------------------%
if isempty(S_type)
    if length(unique(Y)) > 2
        S_type = 'regress1s';
    else
        S_type = 'SVM';
    end
end

if strcmp(S_type(1:3),'qa_') 
    k = S_type(4:end);
    eval(['R.mdl = train' k '(X,Y);']);
    R.Parameter = Parameter;
elseif strcmp(S_type,'test')
    R = [];
else
    R = UMAT(X,Y,C,S_type,M_type,Parameter);
end

R.S_type = S_type;    
%------------------------------------------------------------------------------------------------%
%% Quick Actions for Multiple Machine Learning Algorithms
%------------------------------------------- Regression -----------------------------------------%

%---------------------------------------------------------------- LINEAR MODELS
%    * Regularized Least squares Linear regression (RLR)
%      S_type = 'qa_RLR'
%---------------------------------------------------------------- SPLINES and POLYNOMIALS 
%    * Locally Weighted Polynomials (LWP)    
%      S_type = 'qa_LWP'
%---------------------------------------------------------------- NEIGHBORS
%    * K-nearest Neighbors Regression (KNNR)
%      S_type = 'qa_KNNR'
%    * Weighted K-nearest Neighbors Regression (WKNNR)
%      S_type = 'qa_WKNNR'
%---------------------------------------------------------------- TREE MODELS
%    * Decision Trees (TREE)
%      S_type = 'qa_TREE'
%    * Bagging Trees (BAGTREE)
%      S_type = 'qa_BAGTREE'
%    * Boosting Trees (BOOST)
%      S_type = 'qa_BOOST'
%    * Random Forests (RF1)
%      S_type = 'qa_RF1'
%    * Boosting Random Trees (RF2)
%      S_type = 'qa_RF2'
%---------------------------------------------------------------- NEURAL NETWORS
%    * Neural Networks (NN)
%      S_type = 'qa_NN'
%    * Extreme Learning Machines (ELM)
%      S_type = 'qa_ELM'
%---------------------------------------------------------------- KERNEL METHODS
%    * Support Vector Regression (SVR)
%      S_type = 'qa_SVR'
%    * Kernel Ridge Regression (KRR), aka Least Squares SVM
%      S_type = 'qa_KRR'
%    * Relevance Vector Machine (RVM)
%      S_type = 'qa_RVM'
%    * Kernel Signal to Noise Regression (KSNR)
%      S_type = 'qa_KSNR'
%    * Structured KRR (SKRR)
%      S_type = 'qa_SKRRrbf'
%      S_type = 'qa_SKRRlin'
%    * Random Kitchen Sinks Regression (RKS)
%      S_type = 'qa_RKS'
%---------------------------------------------------------------- GAUSSIAN PROCESSES
%    * Gaussian Process Regression (GPR)
%      S_type = 'qa_GPR'
%    * Variational Heteroscedastic Gaussian Process Regression (VHGPR)
%      S_type = 'qa_VHGPR'
%    * Warped Gaussian Processes (WGPR)
%      S_type = 'qa_WGPR'
%    * Sparse Spectrum Gaussian Process Regression (SSGPR)
%      S_type = 'qa_SSGPR'
%    * Twin Gaussian Processes (TGP)
%      S_type = 'qa_TGP'
%------------------------------------------------------------------------------------------------%
%% Reference
%  https://github.com/IPL-UV
%------------------------------------------------------------------------------------------------%
%   METHODS: Several statistical algorithms are used:
%   --------------------------------------
%
%    * Least squares Linear regression (LR)
%           -- Note that the solution is not regularized
%
%    * Least Absolute Shrinkage and Selection Operator (LASSO).
%           -- This is a Mathworks implementation so you will need the corresponding Matlab toolbox
%           -- We use a 5-fold cross-validation scheme here
%
%    * Elastic Net (ELASTICNET).
%           -- This is a Mathworks implementation so you will need the corresponding Matlab toolbox
%           -- The tradeoff l_1-norm alpha parameter was fixed to 0.5 and could be also crossvalidated
%           -- We use a 5-fold cross-validation scheme here
%
%    * Decision trees (TREE)
%           -- The minimum number of samples to split a node was fixed to 30 and could be also crossvalidated
%           -- The code for doing pruning is commented
%
%    * Bagging trees (BAGTREE)
%           -- The maximum number of trees was set to 200 but could be also crossvalidated
%
%    * Boosting trees (BOOST)
%           -- The maximum number of trees was set to 200 but could be also crossvalidated
%
%    * Neural networks (NN)
%           -- Functions included to automatically train and test standard 1-layer neural
%              networks using the Matlab functions "train" and "sim". The code might not
%              work in newer versions of Matlab, say >2012
%           -- The number of hidden neurons is crossvalidated but no regularization is included
%
%    * Extreme Learning Machines (ELM)
%           -- The standard version of the ELM with random initialization of the weights
%              and pseudoinverse of the output spanning subspace.
%           -- The number of hidden neurons is crossvalidated but no regularization is included
%
%    * Support Vector Regression (SVR)
%           -- Standard support vector implementation for regression and function approximation using the libsvm toolbox.
%           -- Three parameters are adjusted via xval: the regularization term C, the \varepsilon insensitivity
%              tube (tolerated error) and a kernel lengthscale \sigma.
%           -- We include Matlab wrappers for automatic training of the SVR. The
%              wrappers call libsvm compiled functions for training and testing.
%           -- The original source code of libsvm can be obtained from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
%              Please cite the original implementation when appropriate.
%
%           -- We also include our own compilation of the libsvm functions for
%              Linux, Windows and Mac. You are encouraged to use our source and binaries for other
%              platforms in http://www.uv.es/~jordi/soft.htm
%
%             [Smola, 2004] A. J. Smola and B. Sch?lkopf, ※A tutorial on support vector regression,"
%              Statistics and Computing, vol. 14, pp. 199每222, 2004.
%
%    * Kernel Ridge Regression (KRR), aka Least Squares SVM
%           -- Standard least squares regression in kernel feature space.
%           -- Two parameters are adjusted: the regularization term \lambda and an RBF kernel lengthscale \sigma.
%
%    * Relevance Vector Machine (RVM)
%
%           -- We include here the MRVM implementation by Arasanathan Thayananthan (at315@cam.ac.uk)
%              (c) Copyright University of Cambridge
%           -- Please cite the original implementation when appropriate.
%
%              [Thayananthan 2006] Multivariate Relevance Vector Machines for Tracking
%                        Arasanathan Thayananthan et al. (University of Cambridge)
%                        in Proc. 9th European Conference on Computer Vision 2006.
%
%    * Gaussian Process Regression (GPR)
%           -- We consider an anisotropic RBF kernel that has a scale, lengthscale
%              per input feature (band), and a constant noise power parameter as hyperparameters
%           -- The full GP toolbox can be downloaded from http://www.gaussianprocess.org/gpml
%              We include here just two functions "gpr.m" and "minimize.m" in the
%              folder /vhgpr for the sake of convenience.
%           -- Please cite the original implementation when appropriate.
%
%              [Rasmussen 2006] Carl Edward Rasmussen and Christopher K. I. Williams
%                   Gaussian Processes for Machine Learning
%                   The MIT Press, 2006. ISBN 0-262-18253-X.
%
%    * Variational Heteroscedastic Gaussian Process Regression (VHGPR)
%           -- We consider an anisotropic RBF kernel that has a scale, lengthscale
%              per input feature (band), and a input-dependent noise power parameter as hyperparameters
%           -- The original source code can be downloaded from http://www.tsc.uc3m.es/~miguel/
%              Here we include for convenience. If you're interested in VHGPR, please cite:
%
%              [L芍zaro-Gredilla, 2011] M. L芍zaro-Gredilla and M. K. Titsias, "Variational
%                     heteroscedastic gaussian process regression,"
%                     28th International Conference on Machine Learning, ICML 2011.
%                     Bellevue, WA, USA: ACM, 2011, pp. 841每848.
%
%   --------------------------------------
%   NOTE:
%   --------------------------------------
%
%   All the programs included in this package are intended for illustration
%   purposes and as accompanying software for the paper:
%
%           Miguel L芍zaro-Gredilla, Michalis K. Titsias, Jochem Verrelst and
%           Gustavo Camps-Valls. "Retrieval of Biophysical Parameters with
%           Heteroscedastic Gaussian Processes". IEEE Geoscience and Remote
%           Sensing Letters, 2013
%
%   Shall the software is useful for you in other geoscience and remote sensing applications,
%   we would greatly acknowledge citing our paper above. Also, please consider
%   citing these papers for particular methods included herein:
%
%   [KRR, NN]  Nonlinear Statistical Retrieval of Atmospheric Profiles from MetOp-IASI and MTG-IRS Infrared Sounding Data
%     Gustavo Camps-Valls, Jordi Mu?oz-Mar赤, Luis G車mez-Chova, Luis Guanter and Xavier Calbet
%     IEEE Transactions on Geoscience and Remote Sensing, 50(5), 1759 - 1769 2012
%
%   [SVR]  Robust Support Vector Regression for Biophysical Parameter Estimation from Remotely Sensed Images
%     Gustavo Camps-Valls, L. Bruzzone, Jose L. Rojo-?lvarez, Farid Melgani
%     IEEE Geoscience and Remote Sensing Letters, 3(3), 339-343, July 200
%
%   [RVM]  Retrieval of Oceanic Chlorophyll Concentration with Relevance Vector Machines
%     G. Camps-Valls, L. Gomez-Chova, J. Vila-Franc谷s, J. Amor車s-L車pez, J. Mu?oz-Mar赤, and J. Calpe-Maravilla
%     Remote Sensing of Environment. 105(1), 23-33, 2006
%
%   [GPR]  Retrieval of Vegetation Biophysical Parameters using Gaussian Processes Techniques
%     J. Verrelst, L. Alonso, G. Camps-Valls, J. Delegido and J. Moreno
%     IEEE Transactions on Geoscience and Remote Sensing, 50(5), 1832 - 1843. 2012
%
%   [GPR/VHGPR]  Retrieval of Biophysical Parameters with Heteroscedastic Gaussian Processes
%     Miguel L芍zaro-Gredilla, Michalis K. Titsias, Jochem Verrelst and Gustavo Camps-Valls.
%     IEEE Geoscience and Remote Sensing Letters, 2013
    
