%%
% Wine Dataset Classification
% Single Layer FLANN (Functional Link Artificial Neural Network)
% Author: Balasubramanyam Evani
%% 

clear all;
rng('default');
rng(5);

filename = 'pima.csv';
table = readtable(filename);
data = table2array(table);

[rows, cols] = size(data);
indx = randperm(rows);
p = 0.7;
training = data(indx(1:round(p*rows)), :);
test = data(indx(round(p*rows)+1:end), :);

xi = training(:,1:size(data,2)-1);
ti = training(:,size(data,2));

xt = test(:,1:size(data,2)-1);
tt = test(:,size(data,2));

for k=1:size(xi,2)
	xi(:,k) = (xi(:,k) - mean(xi(:,k)))/std(xi(:,k));
	xt(:,k) = (xt(:,k) - mean(xt(:,k)))/std(xt(:,k));
end

N = 5;
epochs = 35000;
eta = 0.51;

weights = randn(size(xi,2)*N,2);

X = zeros(size(xi,1),size(xi,2)*N);
j = 1;	
for i=1:size(xi,2)
	in = fel(xi(:,i));
	X(:,N*(j-1)+1:N*(j-1)+N) = in;
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
	Xt(:,N*(j-1)+1:N*(j-1)+N) = in;
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
	x3 = cos(pi*vec);
	x4 = sin(2*pi*vec);
	x5 = cos(2*pi*vec);
	res = [x1 x2 x3 x4 x5];

end