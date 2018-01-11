% Initialization values
dat = load('assignment3_data');

data = dat.xi;
data = data';

% Training/Testing set 
train_len = length(data) * 0.80;
test_len = length(data) - train_len;

data_train = data(1:train_len,:);
data_test = data(train_len + 1:end,:);

% Generate labels
label = dat.tau;
label_train = label(1:train_len);
label_test = label(train_len + 1:end);

[N, P] = size(dat.xi);

max_epochs = 200;
lr = 0.05;

% Initialize weights
w1 = randi([0 1], 1, N) * 2 - 1;
w1 = w1 ./ norm(w1);
w2 = randi([0 1], 1, N) * 2 - 1;
w2 = w2 ./ norm(w2);

% Maintain errors per epoch
training_errors = [];
testing_errors = [];

for i = 1:max_epochs
    for j = 1:train_len   
        
        % Get random example and the label from training set
        point_idx = randi(size(data_train,1));
        point = data_train(point_idx,:);
        tau = label_train(point_idx);

        % forward prop
        sigma = tanh(w1 * point') + tanh(w2 * point');

        % Backward prop, compute delta weights and adjust weights
        delta_w1 = (sigma - tau) * (1 - tanh(point * w1')^2) * point;
        delta_w2 = (sigma - tau) * (1 - tanh(point * w2')^2) * point;

        w1 = w1 - lr * delta_w1;
        w2 = w2 - lr * delta_w2;
    end
    
    % Compute train and test error after epoch
    error_train = 1 / train_len * sum(((tanh(w1 * data_train') + tanh(w2 * data_train') - label_train).^2) / 2);
    error_test = 1 / test_len * sum(((tanh(w1 * data_test') + tanh (w2 * data_test') - label_test).^2) / 2);
    
    % Store errors for visualization
    training_errors = [training_errors error_train];
    testing_errors = [testing_errors error_test];
end

% Visualize error vs epochs
plot(training_errors)
hold on
plot(testing_errors)
xlabel('Epoch')
ylabel('Error')
title('Error vs epoch')
legend('Train Error','Test Error')

% Plot the first weight vector after convergence
figure;
bar(w1)
xlabel('Weight (weight vector 1)')
ylabel('Value')
title('Weights with their values (after convergence)')

% Plot the second weight vector after convergence
figure;
bar(w2)
xlabel('Weight (weight vector 2)')
ylabel('Value')
title('Weights with their values (after convergence)')