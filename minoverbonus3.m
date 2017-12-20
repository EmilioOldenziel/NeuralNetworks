nd = 50;

alphas = 0.25:0.25:6;

lambda = [0 0.1 0.2 0.3 0.4 0.5];

% generalisation error curve for every N
gen_error = zeros(length(alphas),length(Ns));
for l = lambda
    N
    for a = alphas
        a
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
            
            for lb=1:length(label)
                r = rand(1);
                if r < l
                    label(lb) = -1 * label(lb);
                end
            end

            % Use just as many weights as inputs
            old_weights = weights;
            
            error = zeros(1, P);
            for i = 1:max_epochs
                stability = data * weights' .* label' / norm(weights);
                [val, idx] = min(stability);
                old_weights = weights;
                weights = weights + data(idx,:) .* label(idx) / N;
                diff = norm(abs((weights - old_weights)./old_weights));
                if (diff < 0.1)
                    break;
                end
                %for j = 1:P
%
%                    error(j) = (weights * data(j,:)') * label(j);
%                    if (error(j) <= 0)
%                        weights = weights + 1 / N * data(j,:) * label(j);
%                    end
%                end
%                if all(error > 0)
%                    Q = Q + 1;
%                    break;
%                end
            end
            error = (1 / pi) * acos((weights * ones(1, N)') / ((abs(weights) * abs(ones(1,N))')));
            cumerror = [cumerror error];
        end
        % insert gen_error
        ii = find(lambda == l);
        jj = find(alphas == a);
        gen_error(ii,jj) = mean(cumerror);
    end
end
 
%%

figure;
l = [];
% plot curve for each N
for i=1:1:length(lambda)
    plot(alphas,gen_error(i,:));
    hold on;
    % legenda labels
    l = strvcat(l, ['lambda=' num2str(lambda(1,i))])
end
title(['Generalisation error, n_{d}=' num2str(nd)]);
xlabel('\alpha');
ylabel('generalisation error \epsilon_{g}');
legend(l);
