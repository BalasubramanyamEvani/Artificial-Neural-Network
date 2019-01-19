inputValues = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

targetValues = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    targetValues(labels(n) + 1, n) = 1;
end;

w = load('weights.mat');
hidden_weights = w.hidden_weights;
output_wieghts = w.output_weights;

l1 = inputValues.' * hidden_weights;
act1 = sigmoid(l1);
l2 = act1 * output_wieghts;
y = sigmoid(l2);

correct = 0;
not_correct = 0;

for i=1:size(y)
	[~ , ind] = max(y(i,:));
	ind = ind - 1;
	if ind == labels(i)
		correct = correct + 1;
	else
		not_correct = not_correct + 1;
	end
end

fprintf('correct = %d\n',correct);
fprintf('not_correct = %d\n',not_correct);
acc = correct/(correct+not_correct);
sprintf('acc = %f\n',acc*100)

function [res] = sigmoid(val)
	res = (1+exp(-val)).^-1;
end