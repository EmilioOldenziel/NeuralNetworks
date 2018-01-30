nd = 50;

alphas = 0.25:0.25:12;

kappa_final = [];
kappa_final_var = [];

for a = alphas
    cumkappa = [];
    a
    for run = 1:nd
        % Initialization values
        N = 100;
        P = round(a*N);
        max_epochs = 3000;

        % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
        data = 0 + sqrt(1) * randn(P, N);

        weights = zeros(1, N);
        
        % Generate random P labels being -1 or 1 
        %label = randi([0 1], 1, P) * 2 - 1;
        label = sign(ones(1, N) * data'); 

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
        
        % Calculate kmax after training
        stability = data * weights' .* label' / norm(weights);
        [val, idx] = max(stability);
        cumkappa = [cumkappa val];
    end
    kappa_final = [kappa_final, mean(cumkappa)];
    kappa_final_var = [kappa_final_var, var(cumkappa)];
end

figure;
errorbar(alphas,kappa_final, kappa_final_var);
title(['Maximum stability vs alpha, n_{d}=' num2str(nd) ' N=' num2str(N)]);
xlabel('\alpha');
ylabel('Maximum stability \kappa_{max}');

