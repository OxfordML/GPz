function [x,y,color,counts] = reduce(x,y,color,bins)

    
    mnX = min(x);
    mnY = min(y);
    
    wx = (max(x)-mnX)/bins;
    wy = (max(y)-mnY)/bins;

    xi = floor((x-mnX)/wx)+1;
    yi = floor((y-mnY)/wy)+1;
    
    [row,col,counts] = find(sparse(xi,yi,1));
    
    if(isempty(color))
        color = log(counts);
    else
        [~,~,sums] = find(sparse(xi,yi,color));
        color = sums./counts;
    end
    
    x = (row-1)*wx+wx/2+mnX;
    y = (col-1)*wy+wy/2+mnY;
    
end

