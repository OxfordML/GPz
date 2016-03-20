function [GAMMA,dGAMMA,g_dim] = getGamma(theta,method,m,d)

    switch(method)
        case 'GL'
            g_dim = 1;
            GAMMA = num2cell(theta(m*d+1)*ones(m,1)');
            dGAMMA = 0;
        case 'VL'
            g_dim = m;
            GAMMA = num2cell(theta(m*d+1:m*d+m)');
            dGAMMA = zeros(1,m);
        case 'GD'
            g_dim = d;
            GAMMA = diag(theta(m*d+1:m*d+d));
            if(m>1)
                GAMMA = mat2cell(reshape(repmat(GAMMA(:),m,1),d,d,m),d,d,ones(1,m));
            else
                GAMMA = mat2cell(GAMMA,d,d);
            end
            dGAMMA = zeros(d,1);
        case 'VD'
            g_dim = d*m;
            GAMMA = zeros(d,d,m);
            for j=1:m
                GAMMA(:,:,j) = diag(theta(m*d+d*(j-1)+1:m*d+j*d));
            end
            
            if(m>1)
                GAMMA = mat2cell(GAMMA,d,d,ones(1,m));
            else
                GAMMA = mat2cell(GAMMA,d,d);
            end

            dGAMMA = zeros(d,m);
        case 'GC'
            g_dim = d*d;
            
            if(m>1)
                GAMMA = mat2cell(reshape(repmat(theta(m*d+1:m*d+d*d),m,1),d,d,m),d,d,ones(1,m));
            else
                GAMMA = mat2cell(reshape(theta(m*d+1:m*d+d*d),d,d),d,d);
            end
            
            dGAMMA = zeros(d,d);
        case 'VC'
            g_dim = d*d*m;
            
            if(m>1)
                GAMMA = mat2cell(reshape(theta(m*d+1:m*d+d*d*m),d,d,m),d,d,ones(1,m));
            else
                GAMMA = mat2cell(reshape(theta(m*d+1:m*d+d*d*m),d,d),d,d);
            end
            
            dGAMMA = zeros(d,d,m);
    end
end