function [RX,P] = FRtool(X,method,parameter)
%% Assistant for Feature Reduction.
%------------------------------------------------------------------------------------------------%
% - Z.K.X. 2018/12/12
%------------------------------------------------------------------------------------------------%
%% Dependency
%------------------------------------------------------------------------------------------------%
%% Input
%-- X: independent variable (subjects * variates matrix)  
%-- method: method for feature fusion
%           (1) 'PCA' - Principal Component Analysis (default)
%           (2) 'autoencoder' - Sparse Autoencoder 
%-- parameter: method for feature fusion
%           (1) 'PCA'
%                 - parameter = Cumulative contribution rate reserved [0.8 (default)]
%           (2) 'autoencoder'
%                 - parameter.N, Size of hidden representation of the autoencoder [10 (default)]
%                 - parameter.ETF, Transfer function for the encoder ['logsig' (default) | 'satlin']
%                 - parameter.DTF, Transfer function for the decoder ['logsig' (default) | 'satlin'| 'purelin']
%                 - parameter.L2, The coefficient for the L2 weight regularizer [0.001 (default)]
%                 - parameter.LF, Loss function to use for training ['msesparse' (default)]
%                 - parameter.SP, Desired proportion of training examples a neuron reacts to 
%                                 [0.05 (default) | positive scalar value in the range from 0 to 1]
%                 - parameter.SR, Coefficient that controls the impact of the sparsity regularizer
%                                 [1 (default) | a positive scalar value]
%                 - parameter.TA, The algorithm to use for training the autoencoder ['trainscg' (default)]
%                 - parameter.SD, Indicator to rescale the input data [true (default) | false]
%                 - parameter.GPU, Indicator to use GPU for training [false (default) | true]
%                 - parameter.GPU, Indicator to show the training window [false | true (default)]
%                 - parameter.ME, Maximum number of training epochs [1000 (default) | positive integer value]
%------------------------------------------------------------------------------------------------%
%% Output
%-- RX: variable after dimensionality reduction
%-- P: model parameter
%------------------------------------------------------------------------------------------------%
if (nargin < 2) | isempty(method)
    method = 'PCA';
end

if (nargin < 3) 
    parameter = [];
end

if strcmp(method,'PCA') 
    if isempty(parameter)
        parameter = 0.8;
    end
elseif strcmp(method,'autoencoder') 
    if isempty(parameter)
        parameter.N = 10; parameter.ETF = 'logsig'; parameter.DTF = 'logsig'; parameter.L2 = 0.001;
        parameter.LF = 'msesparse'; parameter.SP = 0.05; parameter.SR = 1; parameter.TA = 'trainscg';
        parameter.SD = true; parameter.GPU = false; parameter.SW = true,parameter.ME = 1000;
    else
        a = fieldnames(parameter);
        if isempty(find(strcmp(a,'N')))
            parameter.N = 10;
        end
        if isempty(find(strcmp(a,'ETF')))
            parameter.ETF = 'logsig';
        end        
        if isempty(find(strcmp(a,'DTF')))
            parameter.DTF = 'logsig';
        end      
        if isempty(find(strcmp(a,'L2')))
            parameter.L2 = 0.001;
        end        
        if isempty(find(strcmp(a,'LF')))
            parameter.LF = 'msesparse';
        end         
        if isempty(find(strcmp(a,'SP')))
            parameter.SP = 0.05;
        end        
        if isempty(find(strcmp(a,'SR')))
            parameter.SR = 1;
        end      
        if isempty(find(strcmp(a,'TA')))
            parameter.TA = 'trainscg';
        end  
        if isempty(find(strcmp(a,'SD')))
            parameter.SD = true;
        end  
        if isempty(find(strcmp(a,'GPU')))
            parameter.GPU = false;
        end        
        if isempty(find(strcmp(a,'SW')))
            parameter.SW = true;
        end        
        if isempty(find(strcmp(a,'ME')))
            parameter.ME = 1000;
        end          
    end
end

%%
if strcmp(method,'PCA')
    [eigvector, latent] = PCA(X);
    sum_latent = cumsum(latent/sum(latent));
    dimension = find(sum_latent>=parameter);
    dimension = dimension(1);
    RX = X*eigvector(:,1:dimension);
    P.eigvector = eigvector; P.eigvalue = latent; P.dimension = dimension;
elseif strcmp(method,'autoencoder')
    X = X';
    P.autoenc = trainAutoencoder(X,parameter.N,'EncoderTransferFunction',parameter.ETF,...
        'DecoderTransferFunction',parameter.DTF,'L2WeightRegularization',parameter.L2,...
        'LossFunction',parameter.LF,'SparsityProportion',parameter.SP,'SparsityRegularization',parameter.SR,...
        'TrainingAlgorithm',parameter.TA,'ScaleData',parameter.SD,'UseGPU',parameter.GPU,...
        'ShowProgressWindow',parameter.SW,'MaxEpochs',parameter.ME);
    XReconstructed = predict(P.autoenc,X);
    P.mseError = mse(X - XReconstructed);
    RX = encode(P.autoenc,X); RX = RX';
end
