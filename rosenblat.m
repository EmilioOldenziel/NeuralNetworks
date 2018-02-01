nd = 50;

alphas = 0.25:0.25:3;

useBias = 0;

Q_list = zeros(1, length(alphas));
Q_list_std = zeros(1, length(alphas));
q_ind = 1;

for a = alphas
    Qs = [];   
    a
    for run = 1:nd
        Q = 0;
        % Initialization values
        N = 20;
        P = round(a*N);
        max_epochs = 100;

        % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
        data = 0 + sqrt(1) * randn(P, N);

        if useBias == 1
            data = [data ones(1, P)'];
            N = N + 1;
        end
        % Generate P labels being -1 or 1 
        label = randi([0 1], 1, P) * 2 - 1;

        % Use just as many weights as inputs
        weights = zeros(1, N);
        error = zeros(1, P);

        for i = 1:max_epochs
            for j = 1:P

                error(j) = (weights * data(j,:)') * label(j);
                if (error(j) <= 0)
                    weights = weights + 1 / N * data(j,:) * label(j);
                end
            end
            if all(error > 0)
                Q = Q + 1;
                break;
            end
        end
        Qs = [Qs Q];
    end

    % list holding fraction of successful runs as a function of a
    Q_list(q_ind) = sum(Qs) ./ nd;
    Q_list_std(q_ind) = std(Qs ./ nd);
    q_ind = q_ind + 1;
end

errorbar(alphas, Q_list, Q_list_std);
xlim([0 4]);
ylim([0 1]);
title(['Fraction of succesful runs as a function of \alpha']);
xlabel('\alpha = P/N')
ylabel('Q')
