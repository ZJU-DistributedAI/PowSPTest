from scipy.sparse import csr_matrix
import math

flags_type = ['not released','included','released']
flags_map = {'nr' : 0,'inc':1 ,'rel' :2}

state_space = ['relevant',  'active']
state_map = {'r': 0,  'a': 1}

action_space = ['adopt', 'override','wait', 'match','release', 'exit']
action_map = {'a': 0, 'o': 1, 'w': 2, 'm': 3, 'r': 4, 'e': 5}

r = state_map['r']
a = state_map['a']
nr = flags_map['nr']
rel = flags_map['rel']
inc = flags_map['inc']

#action state flag return row
row_map = [
    #r:nr,inc,rel a:nr,inc,rel
    [[0,1,2],[0,1,2]], #a
    [[3,3,3],[3,3,3]], #o
    [[4,5,6],[7,8,9]], #w
    [[7,8,9],[-1,-1,-1]], #m
    [[10,-1,-1],[10,-1,-1]], #r
    [[11,11,11],[11,11,11]] #e
]


def get_single(la,lh,state,flag,action,alpha,rs,gamma,cm,ru,vd,k,cutoff):

    def result(row = 0): #python 好像没有switch.. 先这样吧
        if row in [0,1]:
            return [
                [1, 0, r, nr],
                [0, 1, r, nr],
                [0, 0, r, nr]
            ]
        elif row == 2:
            return [
                [1, 0, r, rel],
                [0, 1, r, inc],
                [0, 0, r, rel]
            ]
        elif row == 3:
            return [
                [la-lh, 0, r, nr],
                [la-lh-1, 1, r, nr],
                [la-lh-1, 0, r, nr]
            ]
        elif row == 4:
            return [
                [la+1, lh, r, nr],
                [la, lh+1, r, nr],
                [la, lh, r, nr]
            ]
        elif row == 5:
            return [
                [la+1, lh, r, inc],
                [la, lh+1, r, inc],
                [la, lh, r, inc]
            ]
        elif row == 6:
            return [
                [la+1, lh, r, rel],
                [la, lh+1, r, inc],
                [la, lh, r, rel]
            ]
        elif row == 7 :
            if lh > 6:
                return [
                    [la+1, lh, a, nr],
                    [la-lh, 1, r, nr],
                    [la, lh+1, r, nr],
                    [la, lh, a, nr]
                ]
            else:
                return [
                    [la+1, lh, a, rel],
                    [la-lh, 1, r, nr],
                    [la, lh+1, r, inc],
                    [la, lh, a, rel]
                ]
        elif row == 8:
            return [
                [la+1, lh, a, inc],
                [la-lh, 1, r, nr],
                [la, lh+1, r, inc],
                [la, lh, a, inc]
            ]
        elif row == 9: #return reslut(7,1)
            return [
                [la+1, lh, a, rel],
                [la-lh, 1, r, nr],
                [la, lh+1, r, inc],
                [la, lh, a, rel]
            ]
        elif row == 10:
            return [[la,lh,state,rel]]
        elif row == 11:
            exit_state = [cutoff-1,cutoff-1,len(state_space)-1,len(flags_type)]
            return [exit_state]

    prob_type = [0,0,0,0,0,0,0,1,1,1,2,2]

    def prob(row = 0):
        prob_mat = [
            [alpha,(1-alpha)*(1-rs),(1-alpha)*rs],
            [alpha,gamma*(1-alpha)*(1-rs),(1-gamma)*(1-alpha)*(1-rs),(1-alpha)*rs],
            [1]
        ]
        return prob_mat[prob_type[row]]

    def reward(row = 0):
        if row in [0,2]:
            #return [-cm]*3
            return [-cm-lh]*3
        if row in [4,5,6]:
            return [-cm] *3
        elif row == 1:
            #return [ru-cm] *3
            return [ru-cm-lh] *3
        elif row == 3:
            return [lh+1-cm]*3
        elif row in [7,8,9]:
            return [-cm,lh-cm,-cm,-cm]
        elif row == 10:
            return [0]
        else:
            return [la+vd]

    def tuple_return(row = -1):
        if row == -1:
            return [[la,lh,state,flag]],[1],[-10**9]
        return result(row),prob(row),reward(row)

    row = row_map[action][state][flag]
    if action == action_map['o'] and la <= lh : #same like or
        row = -1
    elif action == action_map['r'] and not (lh <= 6 and lh >1 and la>=1):
        row = -1
    elif action == action_map['e'] and not ( la > lh and la > k):
        row = -1
    elif cutoff -1 in [la,lh] and action in [action_map['w'],action_map['m']]:
        row = -1
    elif action == action_map['m'] and la < lh :
        row = -1
    elif action == action_map['w'] and state == a and la < lh:
        row = -1

    return tuple_return(row)


def get_mat(cutoff = 10,alpha = 0.2,rs=0.1,gamma = 0,cm = 0,vd = 20,ru = 7/8,k=6):
    pre_state = [[] for i in action_space]
    # result state
    result_state = [[] for i in action_space]
    # trans prob
    P_probability = [[] for i in action_space]
    # reward
    R_reward = [[] for i in action_space]

    def pack_index(l):
        index = ((l[0]*cutoff+l[1])*len(state_space)+l[2])*len(flags_type)+l[3]
        return index

    def append_in_list(i,l,result,prob,reward):#action i l-> result with prob and reward
        for j in range(len(result)):
            pre_state[i].append(pack_index(l))
            result_state[i].append(pack_index(result[j]))
            P_probability[i].append(prob[j])
            R_reward[i].append(reward[j])

    for action in range(len(action_space)):
        for la in range(cutoff):
            for lh in range(cutoff):
                for state in range(len(state_space)):
                    for flag in range(len(flags_type)):
                        result,prob,reward = get_single(la,lh,state,flag,action,alpha,rs,gamma,cm,ru,vd,k,cutoff)
                        append_in_list(action,[la,lh,state,flag],result,prob,reward)

    exit_state = [cutoff-1,cutoff-1,len(state_space)-1,len(flags_type)]
    for action in range(len(action_space)):
        append_in_list(action,exit_state,[exit_state],[1],[0])

    P = [csr_matrix((P_probability[i],(pre_state[i],result_state[i])),
                        shape=(cutoff**2*len(flags_type)*len(state_space) +1,cutoff**2*len(flags_type)*len(state_space) +1))
             for i in range(len(action_space))]

    R = [csr_matrix((R_reward[i],(pre_state[i],result_state[i])),
                        shape=(cutoff**2*len(flags_type)*len(state_space) +1,cutoff**2*len(flags_type)*len(state_space) +1))
             for i in range(len(action_space))]

    return P,R
