%%

% Main Code 
% Author: Balasubramanyam Evani

%%

inputValues = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

targetValues = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetValues(labels(n) + 1, n) = 1;
end;

hidden1_size = 500;
hidden2_size = 300;
eta = 0.1;
epochs = 500;
batch_size = 100;

training_size = size(inputValues,2);
input_dim = size(inputValues,1);
output_dim = size(targetValues,1);

hidden1_weights = randn(input_dim, hidden1_size);
hidden2_weights = randn(hidden1_size,hidden2_size);
output_weights = randn(hidden2_size, output_dim);


hidden1_weights = hidden1_weights./size(hidden1_weights,1);
hidden2_weights = hidden2_weights./size(hidden2_weights,1);
output_weights = output_weights./size(output_weights,1);
    

inputValues = inputValues';
targetValues = targetValues';

n = zeros(batch_size);
figure; hold on;

for itr=1:epochs
	
	for k=1:batch_size

		n(k) = floor(rand(1)*training_size + 1);

		% forward propagation
		l1 = inputValues(n(k),:) * hidden1_weights;
		act1 = sigmoid(l1);
		l2 = act1 * hidden2_weights;
		act2 = sigmoid(l2);
		y = sigmoid(act2 * output_weights);

		% back propagation

		delta3 = ( y .* (1 - y) ) .* (y - targetValues(n(k),:));
		delta2 = (act2 .* (1 - act2)) .* (delta3 * output_weights');
		delta1 = (act1 .* (1 - act1)) .* (delta2 * hidden2_weights');

		% weight updation

		output_weights = output_weights - eta*(act2' * delta3);
		hidden2_weights = hidden2_weights - eta*(act1' * delta2);
		hidden1_weights = hidden1_weights - eta*(inputValues(n(k),:)' * delta1);

	end
	error = 0;
	for k=1:batch_size
		in = inputValues(n(k),:);
		target = targetValues(n(k),:);
		error = error + norm(sigmoid(sigmoid(sigmoid(in*hidden1_weights)*hidden2_weights)*output_weights) - target, 2);
	end

	error = error/batch_size;
	plot(itr,error,'*');
	if rem(itr,100) == 0
		disp(error);
	end

end

function [res] = sigmoid(val)
    res = (1+exp(-val)).^-1;
end