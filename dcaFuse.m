function [Ax,Ay,Xs,Ys] = dcaFuse(X,Y,label)

[p,n] = size(X);
if size(Y,2) ~= n
    error('X and Y must have the same number of columns (samples).');
elseif length(label) ~= n
    error('The length of the label must be equal to the number of samples.');
elseif n == 1
    error('X and Y must have more than one column (samples)');
end
q = size(Y,1);

classes = unique(label);
c = numel(classes);
cellX = cell(1,c);
cellY = cell(1,c);
nSample = zeros(1,c);
for i = 1:c
    index = find(label==classes(i));
    nSample(i) = length(index);
    cellX{1,i} = X(:,index);
    cellY{1,i} = Y(:,index);
end

meanX = mean(X,2);  % Mean of all training data in X
meanY = mean(Y,2);  % Mean of all training data in Y

classMeanX = zeros(p,c);
classMeanY = zeros(q,c);
for i = 1:c
    classMeanX(:,i) = mean(cellX{1,i},2);   % Mean of each class in X
    classMeanY(:,i) = mean(cellY{1,i},2);   % Mean of each class in Y
end

PhibX = zeros(p,c);
PhibY = zeros(q,c);
for i = 1:c
    PhibX(:,i) = sqrt(nSample(i)) * (classMeanX(:,i)-meanX);
    PhibY(:,i) = sqrt(nSample(i)) * (classMeanY(:,i)-meanY);
end

clear label index cellX cellY meanX meanY classMeanX classMeanY


artSbx = (PhibX') * (PhibX);   % Artificial Sbx (artSbx) is a (c x c) matrix
[eigVecs,eigVals] = eig(artSbx);
eigVals = abs(diag(eigVals));

maxEigVal = max(eigVals);
zeroEigIndx = find(eigVals/maxEigVal<1e-6);
eigVals(zeroEigIndx) = [];
eigVecs(:,zeroEigIndx) = [];

[~,index] = sort(eigVals,'descend');
eigVals = eigVals(index);
eigVecs = eigVecs(:,index);

SbxEigVecs = (PhibX) * (eigVecs);

cx = length(eigVals);   % Rank of Sbx
for i = 1:cx
    SbxEigVecs(:,i) = SbxEigVecs(:,i)/norm(SbxEigVecs(:,i));
end

SbxEigVals = diag(eigVals);                 % SbxEigVals is a (cx x cx) diagonal matrix
Wbx = (SbxEigVecs) * (SbxEigVals^(-1/2));	% Wbx is a (p x cx) matrix which unitizes Sbx

clear index eigVecs eigVals maxEigVal zeroEigIndx
clear PhibX artSbx SbxEigVecs SbxEigVals

artSby = (PhibY') * (PhibY);	% Artificial Sby (artSby) is a (c x c) matrix
[eigVecs,eigVals] = eig(artSby);
eigVals = abs(diag(eigVals));

maxEigVal = max(eigVals);
zeroEigIndx = find(eigVals/maxEigVal<1e-6);
eigVals(zeroEigIndx) = [];
eigVecs(:,zeroEigIndx) = [];

[~,index] = sort(eigVals,'descend');
eigVals = eigVals(index);
eigVecs = eigVecs(:,index);

SbyEigVecs = (PhibY) * (eigVecs);

cy = length(eigVals);      % Rank of Sby
for i = 1:cy
    SbyEigVecs(:,i) = SbyEigVecs(:,i)/norm(SbyEigVecs(:,i));
end

SbyEigVals = diag(eigVals);                  % SbyEigVals is a (cy x cy) diagonal matrix
Wby = (SbyEigVecs) * (SbyEigVals^(-1/2));    % Wby is a (q x cy) matrix which unitizes Sby

clear index eigVecs eigVals maxEigVal zeroEigIndx
clear PhibY artSby SbyEigVecs SbyEigVals


r = min(cx,cy);	% Maximum length of the desired feature vector

Wbx = Wbx(:,1:r);
Wby = Wby(:,1:r);

Xp = Wbx' * X;  % Transform X (pxn) to Xprime (rxn)
Yp = Wby' * Y;  % Transform Y (qxn) to Yprime (rxn)

Sxy = Xp * Yp';	% Between-set covariance matrix

[Wcx,S,Wcy] = svd(Sxy); % Singular Value Decomposition (SVD)

Wcx = Wcx * (S^(-1/2)); % Transformation matrix for Xp
Wcy = Wcy * (S^(-1/2)); % Transformation matrix for Yp

Xs = Wcx' * Xp;	% Transform Xprime to XStar
Ys = Wcy' * Yp;	% Transform Yprime to YStar

Ax = (Wcx') * (Wbx');	% Final transformation Matrix of size (rxp) for X
Ay = (Wcy') * (Wby');	% Final transformation Matrix of size (rxq) for Y
