nd = 50;

alphas = 0.25:0.25:3;

%useBias = 0;

Ns = [20 50 100 200 500];

pool = gcp();

for N = Ns
    Q_list = zeros(1, length(alphas));
    q_ind = 1;
    for a = alphas
        Q = 0;   
        a
        parfor run = 1:nd
            % Initialization values
            P = round(a*N);
            max_epochs = 500;

            % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
            data = 0 + sqrt(1) * randn(P, N);

            %if useBias == 1
            %    data = [data ones(1, P)'];
            %    N = N + 1;
            %end
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

    plot(alphas, Q_list, 'DisplayName',['N = ' num2str(N)])
    xlim([0 4]);
    ylim([0 1]);
    title('Fraction of succesful runs as a function of \alpha')
    xlabel('\alpha = P/N')
    ylabel('Q')
    legend(gca,'show')
    hold on;
    
end