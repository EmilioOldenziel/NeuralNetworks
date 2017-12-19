nd = 50;

alphas = 0.25:0.25:3;
Q_list = zeros(1, length(alphas));
q_ind = 1;

pool = gcp();

for a = alphas
    Q = 0;   
    a
    parfor run = 1:nd
        % Initialization values
        N = 20;
        P = round(a*N);
        max_epochs = 500;

        % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
        data = 0 + sqrt(1) * randn(P, N);

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
    end
    
    % list holding fraction of successful runs as a function of a
    Q_list(q_ind) = Q / nd;
    q_ind = q_ind + 1;
end

plot(alphas,Q_list)
xlim([0 4]);
ylim([0 1]);
%set(gca,'XTickLabel',[0 0.5 1 1.5 2 2.5 3]);
title('Fraction of succesful runs as a function of \alpha')
xlabel('\alpha = P/N')
ylabel('Q')
