nd = 100;

alphas = 0.25:0.25:6;

kappa_final = [];

for a = alphas
    Q = 0;    
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
        
        stability = data * weights' .* label' / norm(weights);
        [val, idx] = max(stability);
        cumkappa = [cumkappa val];
    end
    kappa_final = [kappa_final, mean(cumkappa)];
end

figure;
plot(alphas,kappa_final);
title(['Maximum stability vs alpha, n_{d}=' num2str(nd) ' N=' num2str(N)]);
xlabel('\alpha');
ylabel('Maximum stability \kappa_{max}');

