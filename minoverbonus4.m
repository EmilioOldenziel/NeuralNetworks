load('fisheriris'); 

X = meas(:,[3,4]); % here we just use the third and forth features 

labels=grp2idx(species); 
labels(labels==2)=-1; 
labels(labels==3)=-1;

% Get 80% of the samples from class 1 for training
% Get 20% of the samples from class 1 for testing
train_ind1 = randperm(50, 0.8*50);
test_ind1 = setdiff(1:50, train_ind1);
train_class1 = X(train_ind1,:);
train_class1_label = labels(train_ind1);
test_class1 = X(test_ind1, :);
test_class1_label = labels(test_ind1);

% Get 80% of the samples from class 2 for training
% Get 20% of the samples from class 2 for testing
train_ind2 = randperm(100, 0.8*100);
test_ind2 = setdiff(1:100, train_ind2);
train_class2 = X(train_ind2 + 50,:);
train_class2_label = labels(train_ind2 + 50);
test_class2 = X(test_ind2 + 50, :);
test_class2_label = labels(test_ind2 + 50);

% Combine training data and testing labels
traindata = [train_class1; train_class2];
trainlabels = [train_class1_label; train_class2_label];

% Combine testing data and testing labels
testdata = [test_class1; test_class2];
testlabels = [test_class1_label; test_class2_label];

% Perform only once
nd = 1;

% The dataset is by default 2D
N = size(X, 2);
P_train = length(traindata);
P_test = length(testdata);

% Bias toggle and max epochs
usebias = 1;
max_epochs = 100;

% Add one dimension to the input vectors and set them to ones.
if usebias == 1
    N = N + 1;
    traindata = [traindata ones(1, P_train)'];
    testdata = [testdata ones(1, P_test)'];
end

% Count how many correctly classified as a function of epochs
totalcorrect = [];

for run = 1:nd
    correct = 0;
    weights = zeros(1, N);

    % Use just as many weights as inputs
    old_weights = weights;

    for i = 1:max_epochs
        correct = 0;
        stability = traindata * weights' .* trainlabels / norm(weights);
        [val, idx] = min(stability);
        old_weights = weights;
        weights = weights + traindata(idx,:) .* trainlabels(idx) / N;
        diff = norm(abs((weights - old_weights)./old_weights));
        if (diff < 0.001)
            break;
        end
        for p = 1:P_test
            if testdata(p,:) * weights' * testlabels(p) > 0
               correct = correct + 1;
            end
        end
        % Store how many correct after each epoch
        totalcorrect = [totalcorrect correct / P_test];
    end
    plot(1:max_epochs, totalcorrect);
    title(['Test accuracy vs epoch']);
    xlabel('epoch');
    ylabel('Test accuracy');
end


