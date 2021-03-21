function [P] = initialize_P(a_vec, b_vec)
n = length(a_vec);
m = length(b_vec);
P = zeros(n,m);
i =1;
j = 1;
r = a_vec(1);
c = b_vec(1);
while ((i <= n) && (j <= m))
    t = min(r,c);
    P(i,j) = t;
    r = r -t;
    c = c - t;
    if r == 0
        i = i+1;
        if i <= n
            r = a_vec(i);
        end
    end
    if c == 0
        j = j+1;
        if j <= m
            c = b_vec(j);
        end
    end
end
end
        
    
