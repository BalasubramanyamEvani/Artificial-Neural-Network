%%
% Wine Dataset Classification - ANN
% 3 layer - input + hidden (3 neurons) + output
% Author: Balasubramanyam Evani
%% 

clear all;
rng('default');
rng(2);

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

input_dim = size(xi,2);
hidden_dim = 6;
output_dim = size(class,1);
eta = 0.1;
epochs = 500;

hidden_weights = randn(input_dim, hidden_dim);
output_weights = randn(hidden_dim, output_dim);

hidden_weights = hidden_weights./size(hidden_weights,2);
output_weights = output_weights./size(output_weights,2);

%figure;hold on;

for itr=1:epochs
	
	for i=1:size(xi,1)
		
		% forward propagation

		l1 = xi(i,:) * hidden_weights;
		act1 = sigmoid(l1);
		l2 = act1 * output_weights;
		y = sigmoid(l2);

		% back propagation

		delta2 = (y .* (1-y)) .* (y - ti(i,:));
		delta1 = (act1 .* (1-act1)) .* (delta2 * output_weights');

		% weights updation

		output_weights = output_weights - eta*(act1' * delta2);
		hidden_weights = hidden_weights - eta*(xi(i,:)' * delta1);

	end
	
	error = 0;
	for i=1:size(xi,1)

		L1 = xi(i,:) * hidden_weights;
		act1 = sigmoid(L1);
		L2 = act1 * output_weights;
		res = sigmoid(L2);
		error = error + norm(res - ti(i,:),2)/size(xi,1);

	end
	%plot(itr,error,'*');
	%if rem(itr,100) == 0
	%	disp(error);
	%end
end

save('weights','hidden_weights','output_weights');

correct = 0;
not_correct = 0;

for i=1:size(xt,1) 
	z1 = xt(i,:) * hidden_weights;
	a1 = sigmoid(z1);
	z2 = a1 * output_weights;
	res = sigmoid(z2);
	[~,ind] = max(res);
	[~,o] = max(tt(i,:));
	if ind == o
		correct = correct + 1;
	else
		not_correct = not_correct + 1;
	end
end

acc = correct/(correct+not_correct);
fprintf('correct= %f\n',correct);
fprintf('not_correct= %f\n',not_correct);
sprintf('accuracy= %f\n',acc*100)

function [res] = sigmoid(val)
	res = (1+exp(-val)).^-1;
end
