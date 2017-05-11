function Psi = fixSx(Sx,n,sdX,method)

    if(isempty(Sx))
        Psi = [];
        return;
    end
    d = length(sdX);

    [~,width,depth] = size(Sx);

    if(depth==n&&n~=1)
        cube=true;
    else
        cube=false;
    end

    switch(method(2))

        case 'C'
            Psi = zeros(d,d,n);
            if(~cube)
                if(width==1)
                    for i=1:n
                        Psi(:,:,i) = (eye(d)*Sx(i))./(sdX'*sdX);
                    end
                else
                    for i=1:n
                        Psi(:,:,i) = diag(Sx(i,:)./sdX.^2);
                    end
                end
            else
                for i=1:n
                    Psi(:,:,i) = Sx(:,:,i)./(sdX'*sdX);
                end
            end
        
        otherwise
            
            if(~cube)
                if(width==1)
                    Psi = bsxfun(@rdivide,repmat(Sx,1,d),sdX.^2);
                else
                    Psi = bsxfun(@rdivide,Sx,sdX.^2);
                end
            else
                Psi = zeros(n,d);
                for i=1:n
                    Psi(i,:) = diag(Sx(:,:,i)./(sdX'*sdX))';
                end
            end
    end
    
end