function omega = getOmega(Y,method,binWidth)

    n = length(Y);
    if(strcmp(method,'balanced'))
        
        minY = min(Y);
        maxY = max(Y);
        
        if(nargin<3)
            binWidth = (maxY-minY)/100;
        end
        
        bins = ceil((maxY-minY)/binWidth);
        centers = minY+(1:bins)'*binWidth-binWidth/2;
        counts = hist(Y,centers);
        [~,ind] = min(Dxy(Y,centers),[],2);
        omega = (max(counts))./(counts(ind))';
    elseif(strcmp(method,'normalized'))
        omega = (1+Y).^-2;
    else
        omega = ones(n,1);
    end
end