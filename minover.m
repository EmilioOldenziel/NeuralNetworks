nd = 100;

alphas = 0.25:0.25:6;

error_final = [];

for a = alphas
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
        % Generate P labels being -1 or 1 
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
        error = (1 / pi) * acos((weights * ones(1, N)') / ((abs(weights) * abs(ones(1,N))')));
        cumerror = [cumerror error];
    end
    error_final = [error_final, mean(cumerror)];
end

figure;
plot(alphas,error_final);
title(['Generalisation error, n_{d}=' num2str(nd) ' N=' num2str(N)]);
xlabel('\alpha');
ylabel('generalisation error \epsilon_{g}');

