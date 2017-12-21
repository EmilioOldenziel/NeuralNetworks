nd = 50;

alphas = 0.25:0.25:6;

Ns = [10 20 50 100 200];

% generalisation error curve for every N
gen_error = zeros(length(alphas),length(Ns));

for N=Ns
    N
    for a = alphas
        a
        cumerror = [];
        for run = 1:nd
            % Initialization values
            P = round(a*N);
            max_epochs = 3000;

            % Generate P datapoints from N-dimensional gaussian (mean = 0, std = 1)
            data = 0 + sqrt(1) * randn(P, N);

            weights = zeros(1, N);
            % Generate P labels being -1 or 1 
            label = sign(ones(1, N) * data'); 

            % Use just as many weights as inputs
            old_weights = weights;

            error = zeros(1, P);
            for i = 1:max_epochs
                for j = 1:P
                
                    error(j) = (weights * data(j,:)') * label(j);
                    if (error(j) <= 0)
                        weights = weights + 1 / N * data(j,:) * label(j);
                    end
                end
                if all(error > 0)
                    break;
                end
            end
            error = (1 / pi) * acos((weights * ones(1, N)') / ((abs(weights) * abs(ones(1,N))')));
            cumerror = [cumerror error];
        end
        % insert gen_error
        ii = find(Ns == N);
        jj = find(alphas == a);
        gen_error(ii,jj) = mean(cumerror);
    end
end

figure;
l = [];
% plot curve for each N
for i=1:1:length(Ns)
    plot(alphas,gen_error(i,:));
    hold on;
    % legenda labels
    l = strvcat(l, ['N=' num2str(Ns(1,i))])
end
title(['Generalisation error, n_{d}=' num2str(nd) ' max epochs=' num2str(max_epochs)]);
xlabel('\alpha');
ylabel('generalisation error \epsilon_{g}');
legend(l);


