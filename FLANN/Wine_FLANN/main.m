%%
% Wine Dataset Classification
% Single Layer FLANN (Functional Link Artificial Neural Network)
% Author: Balasubramanyam Evani
%% 

clear all;
rng('default');
rng(1);

filename = 'wine.csv';
table = readtable(filename);
table(1,:) = [];
data = table2array(table);

[class, ~ , vals] = unique(data(:,1));
mat = eye(max(vals));
encode = mat(vals,:);
data(:,1) = [];
Data = zeros(size(data,1),size(data,2)+size(class,1));
Data(:,1:size(data,2)) = data;
Data(:,size(data,2)+1:size(data,2)+size(class,1)) = encode;

[rows, cols] = size(Data);
indx = randperm(rows);
p = 0.7;
training = Data(indx(1:round(p*rows)), :);
test = Data(indx(round(p*rows)+1:end), :);

xi = training(:,1:size(data,2));
ti = training(:,size(data,2)+1:size(data,2)+size(class,1));

xt = test(:,1:size(data,2));
tt = test(:,size(data,2)+1:size(data,2)+size(class,1));

for k=1:size(xi,2)
	xi(:,k) = (xi(:,k) - mean(xi(:,k)))/std(xi(:,k));
	xt(:,k) = (xt(:,k) - mean(xt(:,k)))/std(xt(:,k));
end

N = 5;
epochs = 500;
eta = 0.1;

weights = randn(size(xi,2)*N,size(class,1));

X = zeros(size(xi,1),size(xi,2)*N);
j = 1;	
for i=1:size(xi,2)
	in = fel(xi(:,i));
	X(:,5*(j-1)+1:5*(j-1)+5) = in;
	j = j + 1;
end

for itr=1:epochs

    for sample=1:size(X,1)
        y = sigmoid(X(sample,:) * weights);
        error = ti(sample,:) - y;
        for i=1:size(error,2)
            e = error(1,i);
            yk = y(1,i);
            weights(:,i) = weights(:,i) + eta * yk*(1-yk)*e * X(sample,:)';
        end
    end
end

save('weights','weights');

correct = 0;
not_correct = 0;

Xt = zeros(size(xt,1),size(xt,2)*N);
j = 1;	
for i=1:size(xt,2)
	in = fel(xt(:,i));
	Xt(:,5*(j-1)+1:5*(j-1)+5) = in;
	j = j + 1;
end

for i=1:size(Xt,1) 
	res = sigmoid(Xt(i,:) * weights);
	[~,ind] = max(res);
	[~,o] = max(tt(i,:));
	if ind == o
		correct = correct + 1;
	else
		not_correct = not_correct + 1;
	end
end

correctness = correct/(correct+not_correct);
fprintf('correct= %f\n',correct);
fprintf('not_correct= %f\n',not_correct);
sprintf('correctness= %f\n',correctness*100)


function [res] = sigmoid(val)
	res = (1+exp(-val)).^-1;
end

function [res] = fel(vec)

	x1 = vec;
	x2 = sin(pi*vec);
	x3 = cos(3*pi*vec);
	x4 = cos(pi*vec);
	x5 = sin(3*pi*vec);
	res = [x1 x2 x3 x4 x5];

end