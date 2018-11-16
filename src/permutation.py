

def kendall_tau(p1, p2):
    n = len(p1)
    total = 0
    for i in range(n):
        for j in range(i):
            if p1[i] < p1[j] and p2[i] > p2[j]:
                total += 1
            elif p1[i] > p1[j] and p2[i] < p2[j]:
                total += 1
    k = (n*(n-1))/2
    return total/k
