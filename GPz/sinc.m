function y = sinc(x)
    % Normalized sinc, with its continuous value at the origin.
    y = ones(size(x));
    i = x ~= 0;
    y(i) = sinpi(x(i))./(pi*x(i));
end