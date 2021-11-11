function FR = FStool(X,Y,C,FS,force)
%% Assistant for Feature Selection 
% 
% - Z.K.X. 2018/05/20
%------------------------------------------------------------------------------------------------%
%% Dependency
%------------------------------------------------------------------------------------------------%
%% Input
%-- X: independent variable (subjects * variates matrix) or gourp 1 (subjects * variates matrix)
%-- Y: dependent variable (subjects * variates matrix) or gourp 2 (subjects * variates matrix)
%-- C: control variable (subjects * 1)
%-- FS: feature selection strategy (allow multiple strategies:n * 3)
%       FS{n,1} = S_type in UMAT
%       FS{n,2} = M_type in UMAT ([]: feature selection without multiple comparison correction )
%       FS{n,3} = Parameter in UMAT
%       FS{1,4} = stragety of combination (1 - union set; 2 - intersection)
%-- force: only pick out some particular feature sets (allow multiple input)
%          example:
%          [1,1] - p = 0.05; all feature
%          [2,1] - p = 0.05; positive feature
%          [2,2] - p = 0.01; positive feature
%          [3,3] - p = 0.005; negative feature
%------------------------------------------------------------------------------------------------%
%% Output
%-- selected_feature: selected feature by given feature selection strategy
%                     ( p level = [0.05 0.01 0.005 0.001 0.0005 0.0001], separately );
%-- selected_feature_r: r value of selected feature 
%-- UMAT_outputs: outputs in UMAT
%-- fs:selected feature set (1 row: all feature; 2 row: positive feature; 3 row: negative feature)
%-- fs_sum:mean value of different selected features
%------------------------------------------------------------------------------------------------%
%%
if (nargin < 3)
    C = [];
end
if (nargin < 4) | isempty(FS) == 1
    FS{1,1} = []; FS{1,2} = []; FS{1,3} = []; FS{1,4}= 1;
end
if (nargin < 5) 
    force = [];
end

for i = 1:size(FS,1)
    R(i).Results = UMAT(X,Y,C,FS{i,1},FS{i,2},FS{i,3});
end

level = [0.05 0.01 0.005 0.001 0.0005 0.0001];

for i = 1:length(R)
    for j = 1:length(level)
        sig_position{i,j} = find(R(i).Results.p_value < level(j));
        sig_position_corr{i,1} = R(i).Results.sig_variate_corrected;
    end
end

sp_corr = sig_position_corr{1};
for i = 1:length(sig_position_corr)
    if FS{1,4} == 1
        sp_corr = union(sp_corr,sig_position_corr{i});
    else
        sp_corr = intersect(sp_corr,sig_position_corr{i});
    end
end

[a,b] = size(sig_position);
sp = sig_position(1,:);
for i = 1:a
    for j = 1:b
        if FS{1,4} == 1
            sp{1,j} = union(sp{1,j},sig_position{i,j});
        else
            sp{1,j} = intersect(sp{1,j},sig_position{i,j});
        end   
    end
end

if isempty(FS{1,2}) == 1
    FR.selected_feature = sp;
else
    FR.selected_feature = sp_corr;
end

k = zeros(1,length(FR.selected_feature));
for i = 1:length(FR.selected_feature)
    if isempty(FR.selected_feature{i}) == 1
        k(i) = 1;
    end
end

FR.selected_feature(k==1) = [];

for i = 1:length(FR.selected_feature)
    FR.selected_feature_r{1,i} = R(1).Results.r_value(FR.selected_feature{i});
end

for i = 1:length(R)
    FR.UMAT_outputs(i) = R(i).Results;
end

%%
for i = 1:length(FR.selected_feature)
    curr{1,i} = X(:,FR.selected_feature{1,i});
    if length(FR.UMAT_outputs) == 1
        pos = FR.selected_feature{1,i}(FR.selected_feature_r{i}>0);
        neg = FR.selected_feature{1,i}(FR.selected_feature_r{i}<0);
        curr{2,i} = X(:,pos); curr{3,i} = X(:,neg);
        FR.selected_feature{2,i} = pos;
        FR.selected_feature{3,i} = neg;
    end
end
if exist('curr')
    FR.fs = curr;
else
    FR.fs = [];
end
    
if length(FR.UMAT_outputs) == 1 & ~isempty(FR.fs)
for i = 1:size(curr,2);
    if ~isempty(curr{2,i})
        FR.fs_sum{2,i} = mean(curr{2,i},2);
    else
        FR.fs_sum{2,i} = [];
    end
    if ~isempty(curr{3,i})
        FR.fs_sum{3,i} = mean(curr{3,i},2);
    else
        FR.fs_sum{3,i} = [];
    end
    FR.fs_sum{1,i} = [FR.fs_sum{2,i},FR.fs_sum{3,i}];
end
end

if ~isempty(force)
    for i = 1:size(force,1)
        sf_curr{i} = FR.selected_feature{force(i,1),force(i,2)};
        sfr_curr{i} = FR.selected_feature_r{1,force(i,2)};
        fs_curr{i} = FR.fs{force(i,1),force(i,2)};
        fssum_curr{i} = FR.fs_sum{force(i,1),force(i,2)};
    end
    FR.selected_feature = sf_curr;
    FR.selected_feature_r = sfr_curr;
    FR.fs = fs_curr;
    FR.fs_sum = fssum_curr;
end

