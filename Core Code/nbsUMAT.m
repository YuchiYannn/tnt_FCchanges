function R = nbsUMAT(X,Y,C,S_type,M,Parameter,label,coordinate)
%% Network-based Statistic Analysis Applied by UMAT
%------------------------------------------------------------------------------------------------%
% - Z.K.X. 2018/06/05
%------------------------------------------------------------------------------------------------%
%% Dependency
% 1.Newman Fast Community Detection Algorithm
%   Brain Connectivity Toolbox (BCT)
%   Download link: http://www.brain-connectivity-toolbox.net/
%------------------------------------------------------------------------------------------------%
%% Reference
% 1.Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). 
%   Network-based statistic: identifying differences in brain networks. 
%   Neuroimage, 53(4), 1197-1207.
% 2.Rosenberg, M. D., Zhang, S., Hsu, W. T., Scheinost, D., 
%   Finn, E. S., Shen, X., ... & Chun, M. M. (2016). 
%   Methylphenidate modulates functional network connectivity to enhance attention. 
%   Journal of Neuroscience, 36(37), 9547-9557.
% 3.Rosenberg, M. D., Finn, E. S., Scheinost, D., Papademetris, X., Shen, X., 
%   Constable, R. T., & Chun, M. M. (2016). 
%   A neuromarker of sustained attention from whole-brain functional connectivity. 
%   Nature neuroscience,19(1), 165-171.
%------------------------------------------------------------------------------------------------%
%% Process
%  1. Perform UMAT first, then get result file 'R' including statistical measures at each edge 
%     independently. 
%  2. The result array would be transformeed to matrix automatically. A threshold for uncorrected 
%     p value should be setted at the link level.
%  3. Identify any components in the adjacency matrix defined by the set of suprathreshold edges. 
%     These are referred to as observed components. Compute the size of each observed component 
%     identified; that is, the number of edges it comprises. 
%  4. Repeat K times steps 1-3, each time randomly permuting members of the two populations and 
%     storing the size of the largest component identified for each permuation. This yields an 
%     empirical estimate of the null distribution of maximal component size. A corrected  p-value 
%     for each observed component is then calculated using this null distribution.
%------------------------------------------------------------------------------------------------%
%% Input
%-- X: cell array of matrixes (sub * 1) 
%      / UMAT input format is also acceptable
%-- Y: double array of target variable (sub * m) or cell array of matrixes (sub * 1) 
%      / UMAT input format is also acceptable 
%-- C; double array of control variable (sub * m)
%-- S_type: 'S_type' in UMAT
%-- M; parameters for permutation test
%      -- M(1) = threshold of p value (default value is 0.05)
%      -- M(2) = the number of permutations (default value is 1000)
%-- Parameter: 'Parameter' in UMAT
%-- label: 'label' in 'displayR'
%-- coordinate: three-dimensional coordinate of brain regions (nodes * 3 double)
%------------------------------------------------------------------------------------------------%
%% Output
%-- Result: results from UMAT
%-- statis_value: r, r2 or t value for each link
%-- p_value: p value for each link
%-- mask: survival links after permutation test (all/pos/neg)
%-- sig: p value of permutation test (all/pos/neg)
%------------------------------------------------------------------------------------------------%
%%
if (nargin < 3)
    C = [];
end
if (nargin < 4) || isempty(S_type) == 1
    if strcmp(class(Y),'cell') & (size(X,1) == size(Y,1))
        S_type = 'paired T';
    elseif strcmp(class(Y),'cell')
        S_type = 'independent T';
    else
        S_type = 'pearson';
    end
end
if (nargin < 5) || isempty(M) == 1
    M(1) = 0.001; M(2) = 1000;
elseif length(M) == 1
    M(2) = 1000;
end
if (nargin < 6) 
    Parameter = [];
end
if (nargin < 7) 
    label = [];
end
%%
if strcmp(class(X),'cell')
    for i = 1:length(X)
        x(i,:) = tri_oneD(X{i});
    end
end
if strcmp(class(Y),'cell')
    for i = 1:length(Y)
        y(i,:) = tri_oneD(Y{i});
    end
end
if exist('x')
    X = x; 
end
if exist('y')
    Y = y;
end
clear x y;
%%    
result = UMAT(X,Y,C,S_type,[],Parameter); 
%%
if isempty(result.r_value) == 1
    if isempty(result.r2_value) ~= 1
        vmat = result.r2_value;
    else
        vmat = result.t_value;
    end
else
    vmat = result.r_value;
end
pmat = result.p_value;
y = length(pmat);
syms x
eqn = x^2-x == 2*y;
solx = solve(eqn,x);
k = double(solx);
k = k(k>0);
for i = 1:k
    curr0 = zeros(1,i);
    curr1 = pmat(1:k-i);
    p2(i,:) = [curr0,curr1];
    pmat(1:k-i) = [];
end
p2 = rot90(fliplr(p2)) + p2;
for i = 1:k
    curr0 = zeros(1,i);
    curr1 = vmat(1:k-i);
    v2(i,:) = [curr0,curr1];
    vmat(1:k-i) = [];
end
v2 = rot90(fliplr(v2)) + v2;
clear curr* eqn  solx vmat x y;  
%% 
pmat = zeros(k); pos = pmat; neg = pmat;
pmat(p2 < M(1)) = 1;
pos(v2 > 0) = 1; neg(v2 < 0) = 1;
pos = pos.*pmat; neg = neg.*pmat;
[COM_pos,CS_pos] = get_components(pos);
for i = 1:length(CS_pos)
    curr = pos(COM_pos==i,COM_pos==i);
    CS_pos(2,i) = sum(sum(triu(curr,1)));
end
[COM_neg,CS_neg] = get_components(neg);
for i = 1:length(CS_neg)
    curr = neg(COM_neg==i,COM_neg==i);
    CS_neg(2,i) = sum(sum(triu(curr,1)));
end
[COM,CS] = get_components(pmat);
for i = 1:length(CS)
    curr = pmat(COM==i,COM==i);
    CS(2,i) = sum(sum(triu(curr,1)));
end
RCOM = COM; RCS = CS; RCOM_pos = COM_pos; RCS_pos = CS_pos; RCOM_neg = COM_neg; RCS_neg = CS_neg;
Rv2 = v2; Rp2 = p2; Results = result; Rpos = pos; Rneg = neg; Rall = pmat;
%%
if strcmp(S_type,'mattest') | strcmp(S_type,'independent T') | strcmp(S_type,'paired T')
    XY = [X;Y];
    for s = 1:M(2)
        co = randperm(size(XY,1));
        c1 = co(1:size(X,1));
        c2 = co(size(X,1)+1:end);
        result = UMAT(XY(c1,:),XY(c2,:),C,S_type,[],Parameter); 
        if isempty(result.r_value) == 1
            if isempty(result.r2_value) ~= 1
                vmat = result.r2_value;
            else
                vmat = result.t_value;
            end
        else
            vmat = result.r_value;
        end
        pmat = result.p_value;
        for i = 1:k
            curr0 = zeros(1,i);
            curr1 = pmat(1:k-i);
            p2(i,:) = [curr0,curr1];
            pmat(1:k-i) = [];
        end
        p2 = rot90(fliplr(p2)) + p2;
        for i = 1:k
            curr0 = zeros(1,i);
            curr1 = vmat(1:k-i);
            v2(i,:) = [curr0,curr1];
            vmat(1:k-i) = [];
        end
        v2 = rot90(fliplr(v2)) + v2;
        pmat = zeros(k); pos = pmat; neg = pmat;
        pmat(p2 < M(1)) = 1;
        pos(v2 > 0) = 1; neg(v2 < 0) = 1;
        pos = pos.*pmat; neg = neg.*pmat;
        [COM_pos,CS_pos] = get_components(pos);
        for i = 1:length(CS_pos)
            curr = pos(COM_pos==i,COM_pos==i);
            CS_pos(2,i) = sum(sum(triu(curr,1)));
        end
        [COM_neg,CS_neg] = get_components(neg);
        for i = 1:length(CS_neg)
            curr = neg(COM_neg==i,COM_neg==i);
            CS_neg(2,i) = sum(sum(triu(curr,1)));
        end
        [COM,CS] = get_components(pmat);
        for i = 1:length(CS)
            curr = pmat(COM==i,COM==i);
            CS(2,i) = sum(sum(triu(curr,1)));
        end
        edgesize_per(s) = max(CS(2,:));
        edgesize_per_pos(s) = max(CS_pos(2,:));
        edgesize_per_neg(s) = max(CS_neg(2,:));
        disp(['Permutation ',num2str(s),' has been finished!']);
    end
else
    for s = 1:M(2)
        co = randperm(size(Y,1));
        result = UMAT(X,Y(co,:),C,S_type,[],Parameter); 
        if isempty(result.r_value) == 1
            if isempty(result.r2_value) ~= 1
                vmat = result.r2_value;
            else
                vmat = result.t_value;
            end
        else
            vmat = result.r_value;
        end
        pmat = result.p_value;
        for i = 1:k
            curr0 = zeros(1,i);
            curr1 = pmat(1:k-i);
            p2(i,:) = [curr0,curr1];
            pmat(1:k-i) = [];
        end
        p2 = rot90(fliplr(p2)) + p2;
        for i = 1:k
            curr0 = zeros(1,i);
            curr1 = vmat(1:k-i);
            v2(i,:) = [curr0,curr1];
            vmat(1:k-i) = [];
        end
        v2 = rot90(fliplr(v2)) + v2;
        pmat = zeros(k); pos = pmat; neg = pmat;
        pmat(p2 < M(1)) = 1;
        pos(v2 > 0) = 1; neg(v2 < 0) = 1;
        pos = pos.*pmat; neg = neg.*pmat;
        [COM_pos,CS_pos] = get_components(pos);
        for i = 1:length(CS_pos)
            curr = pos(COM_pos==i,COM_pos==i);
            CS_pos(2,i) = sum(sum(triu(curr,1)));
        end
        [COM_neg,CS_neg] = get_components(neg);
        for i = 1:length(CS_neg)
            curr = neg(COM_neg==i,COM_neg==i);
            CS_neg(2,i) = sum(sum(triu(curr,1)));
        end
        [COM,CS] = get_components(pmat);
        for i = 1:length(CS)
            curr = pmat(COM==i,COM==i);
            CS(2,i) = sum(sum(triu(curr,1)));
        end
        edgesize_per(s) = max(CS(2,:));
        edgesize_per_pos(s) = max(CS_pos(2,:));
        edgesize_per_neg(s) = max(CS_neg(2,:));
        disp(['Permutation ',num2str(s),' has been finished!']);
    end     
end
%%
edgesize_per = sort(edgesize_per);
edgesize_per_pos = sort(edgesize_per_pos);
edgesize_per_neg = sort(edgesize_per_neg);
Fall = find(RCS(2,:) >= edgesize_per(ceil(M(2)*0.95)));
Fpos = find(RCS_pos(2,:) >= edgesize_per_pos(ceil(M(2)*0.95)));
Fneg = find(RCS_neg(2,:) >= edgesize_per_neg(ceil(M(2)*0.95)));
for i = 1:length(Fall)
    curr = [edgesize_per,RCS(2,Fall(i))];
    curr = sort(curr);
    F = find(curr == RCS(2,Fall(i)));
    sig_all(i) = 1 - (F/(M(2)+1));
    F = find(RCOM == Fall(i));
    curr = zeros(k);
    curr(F,F) = 1;
    mask_all{i} = curr.*Rall; 
end
for i = 1:length(Fpos)
    curr = [edgesize_per_pos,RCS_pos(2,Fpos(i))];
    curr = sort(curr);
    F = find(curr == RCS_pos(2,Fpos(i)));
    sig_pos(i) = 1 - (F/(M(2)+1));
    F = find(RCOM_pos == Fpos(i));
    curr = zeros(k);
    curr(F,F) = 1;
    mask_pos{i} = curr.*Rpos;     
end
for i = 1:length(Fneg)
    curr = [edgesize_per_neg,RCS_neg(2,Fneg(i))];
    curr = sort(curr);
    F = find(curr == RCS_neg(2,Fneg(i)));
    sig_neg(i) = 1 - (F/(M(2)+1));
    F = find(RCOM_neg == Fneg(i));
    curr = zeros(k);
    curr(F,F) = 1;
    mask_neg{i} = curr.*Rneg;    
end
%%
R.Result = Results;
R.statis_value = Rv2;
R.p_value = Rp2;

if exist('sig_all')
    allD = [edgesize_per,RCS(2,Fall)];
    figure1 = figure;
    axes1 = axes('Parent',figure1);
    box(axes1,'on');
    hold(axes1,'all');
    a = hist(allD,20);
    histfit(allD,20);
    xlim(axes1,[0 max(allD)+0.1]);
    line([min(RCS(2,Fall)),min(RCS(2,Fall))],[0,max(a)],'LineWidth',3,'LineStyle','--','Color',[1 0 0]);        
    title({'Null Distribution'},'FontWeight','bold','FontSize',20);
    R.mask_all = mask_all;
    R.sig_all = sig_all;
end
if exist('sig_pos')
    posD = [edgesize_per_pos,RCS_pos(2,Fpos)];
    figure2 = figure;
    axes1 = axes('Parent',figure2);
    box(axes1,'on');
    hold(axes1,'all');
    a = hist(posD,20);
    histfit(posD,20);
    xlim(axes1,[0 max(posD)+0.1]);
    line([min(RCS_pos(2,Fpos)),min(RCS_pos(2,Fpos))],[0,max(a)],'LineWidth',3,'LineStyle','--','Color',[1 0 0]);        
    title({'Null Distribution for Positive Links'},'FontWeight','bold','FontSize',20);
    R.mask_pos = mask_pos;
    R.sig_pos = sig_pos;
end
if exist('sig_neg')
    negD = [edgesize_per_neg,RCS_neg(2,Fneg)];
    figure3 = figure;
    axes1 = axes('Parent',figure3);
    box(axes1,'on');
    hold(axes1,'all');
    a = hist(negD,20);
    histfit(negD,20);
    xlim(axes1,[0 max(negD)+0.1]);
    line([min(RCS_neg(2,Fneg)),min(RCS_neg(2,Fneg))],[0,max(a)],'LineWidth',3,'LineStyle','--','Color',[1 0 0]);        
    title({'Null Distribution for Negative Links'},'FontWeight','bold','FontSize',20);
    R.mask_neg = mask_neg;
    R.sig_neg = sig_neg;
end
if exist('mask_pos')
    a = mask_pos{1};
else
    a = zeros(k);
end
if exist('mask_neg')
    b = mask_neg{1};
else
    b = zeros(k);
end
mask = a+b;
figure4 = figure;
% displayR(Rv2,Rp2,mask,1,'ratio',label);

if (nargin > 7) 
    if exist('a')
        a = mean(a,2);
        cpos = coordinate(a~=0,:);
        lpos = label(a~=0);
        figure5 = figure;
        scatter3(cpos(:,1),cpos(:,2),cpos(:,3),'MarkerFaceColor',[1 0 0]);
        text(cpos(:,1)+2,cpos(:,2)+0.5,cpos(:,3)+0.5,lpos);
        title({'Positive Network Regions'},'FontWeight','bold','FontSize',20);
    end
    if exist('b')
        b = mean(b,2);
        cneg = coordinate(b~=0,:);
        lneg = label(b~=0);
        figure6 = figure;
        scatter3(cneg(:,1),cneg(:,2),cneg(:,3),'MarkerFaceColor',[0 1 1]);
        text(cneg(:,1)+2,cneg(:,2)+0.5,cneg(:,3)+0.5,lneg);
        title({'Negative Network Regions'},'FontWeight','bold','FontSize',20);
    end    
end

if ~isempty(label)
    for i = 1:length(label)
        for j = 1:length(label)
            LAB{i,j}{1} = label{i}; LAB{i,j}{2} = label{j};
        end
    end
    LAB = tri_oneD(LAB);
    for i = 1:length(R.Result.sig_results)
        R.Result.sig_results(i).Node1 = LAB{R.Result.sig_results(i).position}(1);
        R.Result.sig_results(i).Node2 = LAB{R.Result.sig_results(i).position}(2);
    end
end
    
