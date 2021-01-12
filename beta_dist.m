% Beta probability distribution
function r = beta_dist(b)
if b>0 && b<1
    r = 1/beta(10,1.5) * (b^9)*(1-b)^0.5;
else
    r = 0;
end
end