import transmaker1
import mdptoolbox
import sys

esp = 0.01
alpha = 0
rho = 0
res = []
while alpha < 0.5:
    alpha += 0.1
    low = rho
    high = 1
    while high-low > esp :
        rho = (low+high)/2
        mats = transmaker1.TransMat()
        mats.set_attr(rho_ = rho,alpha_ = alpha,cm_ = 0,omega_=0.38,cutoff_=10,rs_=0.1)
        P,R = mats.get_mat()
        vi = mdptoolbox.mdp.PolicyIteration(P, R, 0.999)
        vi.run()
        v0 = vi.V[0]
        if v0 > 0:
            low = rho
        else:
            high = rho
    print(low)
    res.append((alpha,low))
print(res)
