nd = 50;

alphas = 0.25:0.25:6;

error_final = [];

for a = alphas
    Q = 0;    
    cumerror = [];
    a
    for run = 1:nd
        % Initialization values
        N = 20;
        P = round(a*N);
        max_epochs = 1000;

        % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
        data = 0 + sqrt(1) * randn(P, N);

        weights = ones(1, N);
        % Generate P labels being -1 or 1 
        label = sign(ones(1, N) * data'); 

        % Use just as many weights as inputs
        error = zeros(1, P);
        old_weights = weights;

        for i = 1:max_epochs
            %stability = weights ./ abs(weights) * data' .* label;
            stability = (data' .* label)' .* weights / abs(weights);
            [val, idx] = min(stability);
            old_weights = weights;
            weights = weights + 1 / N * data(idx,:) * label(idx);
        end
        error = (1 / pi) * acos((weights * ones(1, N)') / ((abs(weights) * abs(ones(1,N))')));
        cumerror = [cumerror error];
    end
    error_final = [error_final, mean(cumerror)];
end

plot(error_final)
