load('fisheriris'); 

X = meas(:,[3,4]); % here we just use the third and forth features 

labels=grp2idx(species); 
labels(labels==2)=-1; 
labels(labels==3)=-1;

train_ind1 = randperm(75, 0.8*75);
test_ind1 = setdiff(1:75, train_ind1);
train_class1 = X(train_ind1,:);
train_class1_label = labels(train_ind1);
test_class1 = X(test_ind1, :);
test_class1_label = labels(test_ind1);

train_ind2 = randperm(75, 0.8*75);
test_ind2 = setdiff(1:75, train_ind2);
train_class2 = X(train_ind2 + 75,:);
train_class2_label = labels(train_ind2 + 75);
test_class2 = X(test_ind2 + 75, :);
test_class2_label = labels(test_ind2 + 75);

traindata = [train_class1; train_class2];
trainlabels = [train_class1_label; train_class2_label];

testdata = [test_class1; test_class2];
testlabels = [test_class1_label; test_class2_label];

nd = 1;

N = 2;
P_train = length(traindata);
P_test = length(testdata);

usebias = 1;
max_epochs = 100;

if usebias == 1
    N = N + 1;
    traindata = [traindata ones(1, P_train)'];
    testdata = [testdata ones(1, P_test)'];
end

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
        correct
        totalcorrect = [totalcorrect correct / P_test];
    end
    plot(1:max_epochs, totalcorrect);
end


