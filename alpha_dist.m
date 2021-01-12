% Beta probability distribution
function r = alpha_dist(a)
if a>0 && a<1
    r = 1/beta(1.5,10) * (a^0.5)*(1-a)^9;
else
    r = 0;
end
end