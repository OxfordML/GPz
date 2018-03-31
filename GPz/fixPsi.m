function newPsi = fixPsi(Psi,n,sdX,method)

    
    if(isempty(Psi))
        newPsi = [];
        return;
    end
    d = length(sdX);

    [height,width,depth] = size(Psi);
    
    if(height==d&&width==d&&depth==n)
        cube=true;
    else
        cube=false;
    end

    switch(method(2))

        case 'C'
            
            newPsi = zeros(d,d,n);
            if(~cube)
                
                if(width==1)
                    for i=1:n
                        newPsi(:,:,i) = (eye(d)*Psi(i))./(sdX'*sdX);
                    end
                else
                    for i=1:n
                        newPsi(:,:,i) = diag(Psi(i,:)./sdX.^2);
                    end
                end
            else
                for i=1:n
                    newPsi(:,:,i) = Psi(:,:,i)./(sdX'*sdX);
                end
            end
        
        otherwise
            
            if(~cube)
                if(width==1)
                    newPsi = bsxfun(@rdivide,repmat(Psi,1,d),sdX.^2);
                else
                    newPsi = bsxfun(@rdivide,Psi,sdX.^2);
                end
            else
                newPsi = zeros(n,d);
                for i=1:n
                    newPsi(i,:) = diag(Psi(:,:,i)./(sdX'*sdX))';
                end
            end
    end
    
end