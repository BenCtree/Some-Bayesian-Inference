% Uniform dist on (0,1)
function r = omega_dist(w)
if w>0 && w<0.1
    r = 1;
else
    r = 0;
end
end