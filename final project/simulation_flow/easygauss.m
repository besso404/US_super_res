function [Y] = easygauss(X, mu,sigma)
Y = exp(-0.5*((X-mu)./sigma).^2);
end

