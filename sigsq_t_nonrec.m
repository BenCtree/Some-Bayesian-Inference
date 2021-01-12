function sigsqs = sigsq_t_nonrec(a, b, w, t, returns, sigsq_1)
sigsqs(1) = w + a*(returns(1)^2) + b*sigsq_1;
for i = 2:t
    sigsqs(i) = w + a*(returns(t-1)^2) + b*sigsqs(i-1);
    %sigsqs(i) = r;
end
end %end function