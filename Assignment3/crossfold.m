

% Initialization values
dat = load('assignment3_data');

xi = dat.xi;
xi = xi';

tau = dat.tau;
tau = tau'

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
    
    errors = [errors gradientdescentbonus2(xi(train,:), tau(train, :)', xi(test, :), tau(test, :)')];
end

figure;
bar(errors);
hold on;
plot((1:kfold),ones(1,kfold)*mean(errors),'r');
legend('test error', 'mean test error')
