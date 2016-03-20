function D = Dxy( X, Y)

xx = sum(power(X,2),2);
yy = sum(power(Y,2),2)';
yb = X*Y';

D = abs(abs(bsxfun(@plus, yy, bsxfun(@minus, xx, 2*yb))));

end

