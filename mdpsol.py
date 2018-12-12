import mdptoolbox
import numpy as np
import transmaker
import ethtransmat as eth

esp = 0.04
def get_rho(alpha, init_high=1, init_low=0, cutoff=7, rs=0.1, omega=0.1, cm=0):
    total_time = 0
    low = init_low
    high = init_high
    rho = (low + high) / 2
    while high-low > esp :
        mats = transmaker.TransMat()
        mats.set_attr(rho_ = rho,alpha_ = alpha,cm_ = cm,omega_=omega,cutoff_= cutoff,rs_=rs)
        P,R = mats.get_mat()
        vi = mdptoolbox.mdp.PolicyIteration(P, R, 0.999)
        vi.run()
        v0 = vi.V[1]
        if v0 > 0:
            low = rho
        else:
            high = rho
        rho = (low+high)/2
        total_time += vi.time
    return rho,total_time

esp2 = 0.5
def get_vd(alpha = 0.1,cm = 0.0,k = 6,rs = 0.1,gamma = 0,cutoff=10,init_high = 10**9,omega=0,returnpolicy = False,ethmat = False):
    low = 0
    high = init_high
    vd = (high+low)/2
    while high-vd > esp2:
        if not ethmat:
            mats = transmaker.TransMat()
            mats.set_attr(alpha_=alpha,vd_=vd,cm_=cm,k_=k,gamma_=gamma,omega_=omega,rs_=rs, cutoff_=cutoff)
            P,R = mats.get_mat(make_reward_type=1,with_exit=True)
        else:
            P,R = eth.get_mat(cutoff=cutoff,k=k,rs=rs,gamma=gamma,cm=cm,vd = vd,alpha=alpha)

        vi = mdptoolbox.mdp.PolicyIteration(P,R,0.999)
        vi.run()
        v0 = vi.V[1]
        exit_state = False
        if not ethmat:
            poli = [i for i in vi.policy]
            for i in range(len(poli)):
                la = int(i/3/cutoff/cutoff)
                lh = int(i/3/cutoff)%cutoff
                #be = int(i/3)%cutoff
                #s = i%3
                flag =  la > lh and la > k #貌似没什么必要判断
                if flag and poli[i] == 4:
                    exit_state = True
                    break
        else:
            for i in vi.policy:
                if i==5:
                    exit_state= True
                    break
        if exit_state and v0 > 0:
            high = vd
        else:
            low = vd
        vd = (low+high)/2
        #print(vd)
    if returnpolicy:
        return vd,vi.policy
    return vd


def get_figure2(cutoff = 7):
    rho = 0
    alphas = [i/20 for i in range(1,10)]
    rhos_1 = []
    rhos_10 = []
    #rs = 0.1
    for alpha in alphas:
        rho ,time = get_rho(alpha,init_low=rho,cutoff=cutoff, rs = 0.1)
        print(alpha,rho)
        rhos_10.append(rho)

    print(alphas,rhos_10)

    rho = 0
    for alpha in alphas:
        rho ,time = get_rho(alpha,init_low=rho,cutoff=cutoff,rs =0.01)
        print(alpha,rho)
        rhos_1.append(rho)

    print(alphas,rhos_1)
    return alphas,rhos_1,rhos_10


def get_figure3(cutoff = 9):
    rs = [i/20 + 0.01 for i in range(0,10)]
    rhos01 = []
    rhos03 = []
    for rs_ in rs:
        rho1,time = get_rho(0.1,rs=rs_,cutoff=cutoff,omega=0)
        rho2,time = get_rho(0.3,rs=rs_,cutoff=cutoff,omega=0)
        print(rs_,rho1)
        print(rs_,rho2)
        rhos01.append(rho1)
        rhos03.append(rho2)

    print(rs,rhos01,rhos03)
    return rs,rhos01,rhos03

def get_figure4(cutoff = 7):
    dx, dy = 0.03, 0.03
    y, x = np.mgrid[slice(0, 0.5 , dy),
                    slice(0, 0.5 , dx)]

    z = x.copy()
    for i in range(len(x)):
        for j in range(len(x)):
            t ,time = get_rho(x[i][j],omega=y[i][j],cutoff=cutoff,rs= 0.1)
            z[i][j] = t
            print(t)
    return x,y,z


def get_figure5(cutoff = 8,k=4,alpha = 0.2,cm = 0.2,gamma = 0,omega=0,rs = 0.041):
    #TODO
    return 0

def get_figure6(cutoff = 12):
    alphas = [i/20 for i in range(1,9)]
    vds0 = []
    vdsa = []
    vdsd = []
    vd0 = 10**9
    vda = 10**9
    for alpha in alphas:
        vd0 = get_vd(alpha=alpha,cm=0,rs=0.0041,gamma=0,cutoff=cutoff,init_high=vd0)
        vda = get_vd(alpha=alpha,cm=alpha,rs=0.0041,gamma=0,cutoff=cutoff,init_high=vda)
        vdd = vda-vd0
        print(vda,vd0,vdd)
        vds0.append(vd0)
        vdsa.append(vda)
        vdsd.append(vdd)

    print(alphas,vds0,vdsa,vdsd)
    return alphas,vds0,vdsa,vdsd

def get_figure7(cutoff=12):
    rss = [i/20 for i in range(1,10)]
    vd1 = 10**9
    vd3 = 10**9
    vd1s = []
    vd3s = []
    for rs in rss:
        vd1 = get_vd(alpha=0.1,cm=0.1,rs=rs,init_high=vd1,cutoff=cutoff) #cm ???
        vd3 = get_vd(alpha=0.3,cm=0.1,rs=rs,init_high=vd3,cutoff=cutoff)
        vd1s.append(vd1)
        vd3s.append(vd3)
        print(vd1,vd3)

    print(rss,vd1s,vd3s)
    return rss,vd1s,vd3s


def get_table3(cutoff = 10):

    action_space_f = ['a', '*','w', '*', 'e']
    action_map = {'a': 0, 'o': 1, 'w': 2, 'm': 3, 'e': 4}
    state_space = ['relevant', 'irrelevant', 'active']
    state_map = {'r': 0, 'i': 1, 'a': 2}

    mats = transmaker.TransMat()
    mats.set_attr(alpha_=0.3,vd_=19.5,cm_=0.3,k_=6,gamma_=0,omega_=0,rs_=0.0041, cutoff_=cutoff)
    P,R = mats.get_mat(make_reward_type=1,with_exit=True)

    vi = mdptoolbox.mdp.PolicyIteration(P,R,0.999)
    vi.run()
    poli = vi.policy
    ans = [[['*','*','*'] for j in range(9)] for i in range(9)]
    for i in range(len(poli)):
        la = int(i/3/cutoff/cutoff)
        lh = int(i/3/cutoff)%cutoff
        be = int(i/3)%cutoff
        s = i%3
        if s != 2:
            sp = 1-s #和paper的表顺序有点不一样。。。
            if la <9 and lh <9 and be ==0: #只写9以前的 因为w=0 所以be=0
                flag = True
                if lh == 0 and sp != 0: #lh=0的时候状态只有i
                    flag = False
                if poli[i] == action_map['a'] and la != 0 and sp == 0:
                    flag = False #从左边转移过来只有r
                if lh>0:
                    for j in range(lh) :
                        if 'a' in ans[la][j] and (la,lh,sp) != (0,1,1) : #如果前面adpot就没后面的事情了(转移到010,i了)
                            flag = False
                        if 'e' in ans[la][lh-1] and sp == 1:
                            flag = False #左边退出了不可能达到r状态
                for j in range(la):
                    if 'e' in ans[j][lh]:#如果前面exit就没有后面的事情了
                        flag = False
                if poli[i] == action_map['e'] and sp ==1:
                    flag =False #exit 一般在la+1后执行，现在的状态是i
                if flag:
                    ans[la][lh][sp] = action_space_f[poli[i]]

    print("  | 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 ")
    for i in range(len(ans)):
        print(str(i)+' |'+','.join([''.join(j) for j in ans[i]]))

def get_figure8(cutoff = 10):
    gammas = [0,0.5,1]
    alphas = [i/20 for i in range(1,10)]
    ks = [1,2,4,6,8]
    k_dict = {1:0,2:1,4:2,6:3,8:4}

    rs = 0.0041
    #cm = alpha
    omega = 0
    p0 = [[] for i in ks]
    p5 = [[] for i in ks]
    p1 = [[] for i in ks]
    gamma = 0
    for alpha in alphas:
        for k in ks:
            vd = get_vd(alpha=alpha,cm=alpha,k=k,rs=rs,gamma=gamma,cutoff=cutoff,omega=omega)
            print(vd)
            p0[k_dict[k]].append(vd)
    gamma = 0.5
    for alpha in alphas:
        for k in ks:
            vd = get_vd(alpha=alpha,cm=alpha,k=k,rs=rs,gamma=gamma,cutoff=cutoff,omega=omega)
            print(vd)
            p5[k_dict[k]].append(vd)
    gamma = 1
    for alpha in alphas:
        for k in ks:
            vd = get_vd(alpha=alpha,cm=alpha,k=k,rs=rs,gamma=gamma,cutoff=cutoff,omega=omega)
            print(vd)
            p1[k_dict[k]].append(vd)

    print(p0,p5,p1)
    return gammas,alphas,ks,p0,p5,p1


def get_figure9(cutoff = 8):
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(0, 0.5 , dy),
                    slice(0, 0.5 , dx)]

    z = x.copy()
    for i in range(len(x)):
        for j in range(len(x)):
            vd = get_vd(alpha=x[i][j],omega=y[i][j],cutoff=cutoff,rs= 0.0041,gamma=0,cm=0,init_high=10**8)
            z[i][j] = vd
            print(x[i][j],y[i][j],vd)
    return x,y,z


def get_figure10(cutoff = 9):
    #TODO
    vde = get_vd(alpha=0.2,gamma=0,rs=0.068,cm=0,omega=0,k=6,cutoff=cutoff,ethmat=True)
    print(vde)
    vde = get_vd(alpha=0.3,gamma=0,rs=0.068,cm=0,omega=0,k=6,cutoff=cutoff,ethmat=True)
    print(vde)

