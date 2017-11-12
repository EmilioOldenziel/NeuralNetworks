nd = 50;

alphas = 0.1:0.05:3;
Q_list = zeros(1, length(alphas));
q_ind = 1;

for a = alphas
    Q = 0;    
    for run = 1:nd
        % Initialization values
        N = 20;
        P = int8(a*N);
        max_epochs = 100;

        % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
        data = 0 + sqrt(1) * randn(P, N);

        % Generate P labels being -1 or 1 
        label = randi([0 1], 1, P) * 2 - 1;

        % Use just as many weights as inputs
        weights = zeros(1, N);
        error = zeros(1, P);
        
        for i = 1:max_epochs
            count = 0;
            for j = 1:P

                error(j) = (weights * data(j,:)') * label(j);
                if (error(j) <= 0)
                    weights = weights + 1 / N * data(j,:) * label(j);
                else
                    count = count + 1;
                end
            end
            
            % Successful
            if count == P
                Q = Q + 1;
                break
            end
        end
    end
    
    % list holding fraction of successful runs as a function of a
    Q_list(q_ind) = Q / nd;
    q_ind = q_ind + 1;
end

plot(Q_list)
set(gca,'XTickLabel',[0 0.5 1 1.5 2 2.5 3]);
title("Fraction of succesful runs as a function of a")
xlabel("a = P/N")
ylabel("Q")
