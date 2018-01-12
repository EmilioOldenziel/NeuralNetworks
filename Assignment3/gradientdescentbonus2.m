

% Initialization values
dat = load('../assignment3_data');

data_xi = dat.xi;
data_xi = data_xi';

data_tau = dat.tau;
data_tau = data_tau';

kfold = 5;

p = size(xi,1)
indices = crossvalind('Kfold', p, kfold)
errors = [];

%create folds of training and test sets
for k = 1:kfold
    k
    % test sets
    test = (indices == k);
    train = ~test;

    data_train = data_xi(train,:);
    label_train = data_tau(train, :)';
    data_test = data_xi(test, :);
    label_test = data_tau(test, :)';
    train_len = length(data_train);
    test_len = length(data_test);

    [P, N] = size(data_train);

    max_epochs = 100;
    lr = 0.001;

    w1 = randi([0 1], 1, N) * 2 - 1;
    w1 = w1 ./ norm(w1);
    w2 = randi([0 1], 1, N) * 2 - 1;
    w2 = w2 ./ norm(w2);

    training_errors = [];
    testing_errors = [];

    amount_train_per_epoch = 1;

    for i = 1:max_epochs
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

        error_test = 1 / test_len * sum(((tanh(w1 * data_test') + tanh (w2 * data_test') - label_test).^2) / 2);

    end
    
    errors = [errors error_test];
end

figure;
bar(errors);
xlabel('fold')
ylabel('test error')
hold on;
mean_error = mean(errors)
plot((1:kfold),ones(1,kfold)*mean_error,'r');
legend('test error', 'mean test error')
