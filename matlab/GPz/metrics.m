function scores = metrics(y,mu,sigma,fun)
    
    
    n = length(y);
    [~,order] = sort(sigma);
    
    y = y(order);
    sigma = sigma(order);
    mu = mu(order);

    scores = cumsum(fun(y,mu,sigma))./(1:n)';

end

