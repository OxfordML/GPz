function prior = getPrior(X,Sx,theta,model,set)

m = model.m;

prior = ones(1,m)/m;

for iter=1:100
    old_prior = prior;
    
    [~,~,~,N] = getPHI(X,Sx,theta,model,set);
    
    w = bsxfun(@times,N,prior);
    w = bsxfun(@rdivide,w,sum(w,2));
    
    prior = mean(w);
    
    if(norm(old_prior-prior)/norm(old_prior+prior)<1e-10)
        break
    end
end

end
