k_pos = find(labels == 1)
k_neg = find(labels == -1)

figure;
hold on;
scatter(X(k_pos,1), X(k_pos, 2))
scatter(X(k_neg,1), X(k_neg, 2))

title('feature 1')
xlabel('feature 1')
ylabel('feature 2')
title('Feature space')