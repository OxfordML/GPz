function [centers,means,stds] = bin(x,y,bins)
    
    if(nargin==2)
        bins = 100;
    end
    
    centers = linspace(min(x),max(x),bins)';
    n = length(x);
    m = length(centers);
    [~,id] = min(Dxy(x,centers),[],2);
    counts = full(sum(sparse(1:length(x),id,1,n,m)));
    sums = full(sum(sparse(1:length(x),id,y,n,m)));
    
    remove = counts==0;
    counts(remove) = 1;
    
    means = sums./counts;
    
    sumssqrs = full(sum(sparse(1:length(x),id,(y-means(id)').^2,n,m)));
    
    stds = sqrt(sumssqrs./counts);
    
    means(remove) = [];
    stds(remove) = [];
    centers(remove) = [];
    
end

