% Initialization values
dat = load("assignment3_data");

data = dat.xi;
data = data';

train_len = length(data) * 0.80;
test_len = length(data) - train_len;

data_train = data(1:train_len,:);
data_test = data(train_len + 1:end,:);

label = dat.tau;
label_train = label(1:train_len);
label_test = label(train_len + 1:end);

[N, P] = size(dat.xi);

% XOR Validation
% data = [0 0; 0 1; 1 0; 1 1]
% label = [0 1 1 1]'
% N = 2
% P = 4

max_epochs = 100;
lr = 0.1;
decay = 0.001;

w1 = randi([0 1], 1, N) * 2 - 1;
w1 = w1 ./ norm(w1);
w2 = randi([0 1], 1, N) * 2 - 1;
w2 = w2 ./ norm(w2);

training_errors = [];
testing_errors = [];

amount_train_per_epoch = 1;

for i = 1:max_epochs
    lr = lr * 1/(1 + decay * i)
    i
    for j = 1:train_len        
        point_idx = randi(size(data_train,1));
        point = data_train(point_idx,:);
        tau = label_train(point_idx);

        % forward prop
        sigma = tanh(w1 * point') + tanh(w2 * point');

         % backprop    
        delta_w1 = (sigma - tau) * (1 - tanh(point * w1')^2) * point;
        delta_w2 = (sigma - tau) * (1 - tanh(point * w2')^2) * point;

        w1 = w1 - lr * delta_w1;
        w2 = w2 - lr * delta_w2;
    end
    
    error_train = 1 / train_len * sum(((tanh(w1 * data_train') + tanh(w2 * data_train') - label_train).^2) / 2);
    error_test = 1 / test_len * sum(((tanh(w1 * data_test') + tanh (w2 * data_test') - label_test).^2) / 2);
    
    training_errors = [training_errors error_train];
    testing_errors = [testing_errors error_test];
end

plot(training_errors)
hold on
plot(testing_errors)
xlabel('Epoch')
ylabel('Error')
title('Error vs epoch')
legend('Train Error','Test Error')

