nd = 100;

alphas = 0.25:0.25:1;

lambda = [0 0.1 0.2];

lambda_ind = 1;

error_final = zeros(1, length(lambda));

for l = lambda
    for a = alphas
        a
        Q = 0;    
        cumerror = [];
        for run = 1:nd
            % Initialization values
            N = 100;
            P = round(a*N);
            max_epochs = 3000;

            % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
            data = 0 + sqrt(1) * randn(P, N);

            weights = zeros(1, N);

            % Generate labels such that they are linearly seperable
            % By the teacher perceptron.
            label = sign(ones(1, N) * data'); 

            for lb=1:length(label)
                r = rand(1);
                if r < l
                    label(lb) = -1 * label(lb);
                end
            end

            % Use just as many weights as inputs
            old_weights = weights;

            for i = 1:max_epochs
                stability = data * weights' .* label' / norm(weights);
                [val, idx] = min(stability);
                old_weights = weights;
                weights = weights + data(idx,:) .* label(idx) / N;
                diff = norm(abs((weights - old_weights)./old_weights));
                if (diff < 0.1)
                    break;
                end
            end
            error = (1 / pi) * acos((weights * ones(1, N)') / ((abs(weights) * abs(ones(1,N))')));
            cumerror = [cumerror error];
            
        end
        error_final(lambda_ind) = mean(cumerror)
        lambda_ind = lambda_ind + 1;
    end
    plot(alphas,error_final);
    title(['Generalisation error, n_{d}=' num2str(nd) ' N=' num2str(N)]);
    xlabel('\alpha');
    ylabel('generalisation error \epsilon_{g}');
    legend(gca,'show')
    hold on;
end