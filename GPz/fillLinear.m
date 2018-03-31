function X = fillLinear(X,mu,Sigma)
    
    [n,d] = size(X);
    
    missing = isnan(X);
    
    list = true(n,1);
    groups = logical([]);

    while(sum(list)>0)
        first = find(list,1);
        group = false(n,1);
        group(list) = sum(abs(bsxfun(@minus,missing(list,:),missing(first,:))),2)==0;
        groups = [groups group];
        list(group)=false;
    end
    
    for i=1:size(groups,2)

        group = groups(:,i);

        u = missing(find(group,1),:);
        o = ~u;
        
        if(sum(u)>0)
            Delta = bsxfun(@minus,X(group,o),mu(o));
            X(group,u) = bsxfun(@plus,Delta*(Sigma(o,o)\Sigma(o,u)),mu(u));
        end
    end
    
end