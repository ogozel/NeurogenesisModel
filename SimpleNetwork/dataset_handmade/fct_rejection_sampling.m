% Implement the Von Mieses-Fisher distribution (part 1/2)

% n : number of points we sample
% kappa : concentration parameter (the larger it is, the highest the
% concentration around the mean point, if kappa=0 uniform distribution)
% m : dimension of the system (hypersphere of dimension m-1)
% y: vector of w's that satisfy the rejection sampling scheme


function y = fct_rejection_sampling(n, kappa, m)

dim = m-1;
b = dim / (sqrt(4*kappa*kappa + dim*dim) + 2*kappa);
x = (1-b) / (1+b);
c = kappa*x + dim*log(1-x*x);

y = NaN(n,1);
for i=1:n
    done = false;
    while ~done
        z = betarnd(dim/2,dim/2);
        w = (1 - (1+b)*z) / (1 - (1-b)*z);
        u = rand;
        if kappa*w + dim*log(1-x*w) - c >= log(u)
            done = true;
        end  
    end
    y(i) = w;
end

end