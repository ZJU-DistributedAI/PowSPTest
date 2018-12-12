from scipy.sparse import csr_matrix
import math
import sys
# M := <S,A,P,R>
# we need a transitions matrix: action * (state space *la * lh  * be)^2
# it's too large to apply in mdp
# so we use scipy.sparse matrix

# A
action_space = ['adopt', 'override','wait', 'match', 'exit']
action_map = {'a': 0, 'o': 1, 'w': 2, 'm': 3, 'e': 4}
# S
state_space = ['relevant', 'irrelevant', 'active']
state_map = {'r': 0, 'i': 1, 'a': 2}


class TransMat:
    def __init__(self):
        self.alpha = 0.3  # total mining power of adversary
        self.omega = 0.4  # mining power of eclipsed victim
        self.cm = self.alpha  # expected mining costs of adversary
        self.k = 6  # the confirmation num
        self.gamma = 0  # propagation parameter
        self.rs = 0.1  # stale block rate
        #self.alpha_ = self.alpha/(1-self.omega)  # implicitly! isolate the victim so increase alpha
        self.cutoff = 10  # just cutoff
        self.vd = 1
        self.rho = 0.5

    def set_attr(self,alpha_= 0.3,omega_= 0.38,cm_= 0,k_= 6,gamma_= 0,rs_=0.1,cutoff_=10,vd_= 1,rho_= 0.5):
        self.alpha = alpha_
        self.omega = omega_
        self.cm = cm_
        self.k = k_
        self.gamma = gamma_
        self.rs = rs_
        self.cutoff = cutoff_
        self.vd = vd_
        self.rho = rho_

    def print_attr(self):
        print("alpha,omega,cm,k,gamma,rs,cutoff,vd,rho:")
        print(self.alpha,self.omega,self.cm,self.k,self.gamma,self.rs,self.cutoff,self.vd,self.rho)

    def make_reward_(self,la,lh,be,s,a):
        def wrho(ra,rh):
            return (1-self.rho)*ra-self.rho*rh

        if a == 0:
            return [wrho(-self.cm,lh)]*4
        if a == 1:
            return [wrho(math.floor((lh+1)*(la-be)/la) - self.cm,0)]*4
        if a == 2:
            return [wrho(-self.cm,0)]*4
        if a == 3:
            if la != 0:
                return [wrho(-self.cm,0),wrho(-self.cm,0),wrho(math.floor(lh*(la-be)/la)-self.cm,0),wrho(-self.cm,0),wrho(-self.cm,0)]
            else:
                return [wrho(-self.cm,0),wrho(-self.cm,0),wrho(-self.cm,0),wrho(-self.cm,0),wrho(-self.cm,0)]

        if a == 4:#其实没有用到
            return [la-be+self.vd]

    def make_reward_1(self,la,lh,be,s,a):

        if a == 0:
            return [-self.cm - lh]*4
        if a == 1:
            return [math.floor((lh+1)*(la-be)/la) - self.cm]*4
        if a == 2:
            return [-self.cm]*4
        if a == 3:
            if la != 0:
                return [-self.cm,-self.cm,math.floor(lh*(la-be)/la)-self.cm,-self.cm,-self.cm]
            else:
                return [-self.cm,-self.cm,-self.cm,-self.cm,-self.cm]
        if a == 4:
            return [la-be+self.vd]

    def get_mat(self , make_reward_type =0, with_exit = False):
        #self.print_attr()
        make_reward = self.make_reward_
        if make_reward_type == 1:
            make_reward = self.make_reward_1
        # start state : state_space * la * lh * be
        pre_state = [[] for i in action_space]
        # result state
        result_state = [[] for i in action_space]
        # trans prob
        P_probability = [[] for i in action_space]
        # reward
        R_reward = [[] for i in action_space]

        def pack_index(l,s):
            index = len(state_space)*(l[0]*self.cutoff*self.cutoff+l[1]*self.cutoff+l[2])+s
            return index

        def append_in_list(i,l,s,result,prob,reward):#action i l,s(last state)->result with prob and reward
            for j in range(len(result)):
                pre_state[i].append(pack_index(l,s))
                result_state[i].append(pack_index(result[j][0:3],result[j][3]))
                P_probability[i].append(prob[j])
                R_reward[i].append(reward[j])

        exit_state = [self.cutoff-1,self.cutoff-1,self.cutoff-1,len(state_space)] #set last state to exit
        for la in range(self.cutoff ):
            for lh in range(self.cutoff):
                for be in range(self.cutoff):
                    for s in range(len(state_space)):

                        l = [la,lh,be]
                        cut = False
                        if self.cutoff-1 in l:
                            cut = True

                        def set_unable(i):  #never a policy would include this
                            append_in_list(i, l, s, [[la, lh, be, s]], [1], [-10**9])

                        # index 0 for adopt see P9 table row 1
                        result_0 = [
                            [1,0,0,state_map['i']],
                            [1,0,1,state_map['i']],
                            [0,1,0,state_map['r']],
                            [0,0,0,state_map['i']]
                        ]
                        prob_0 = [self.alpha,
                                  self.omega,
                                  (1-self.alpha-self.omega)*(1-self.rs),
                                  (1-self.alpha-self.omega)*self.rs]
                        reward_0 = make_reward(la,lh,be,s,0)
                        append_in_list(0,l,s,result_0,prob_0,reward_0)
                        # row1 end

                        # index 1 for override ,row 2
                        if la > lh :
                            override_be = be - math.ceil((lh +1)*be/la)
                            result_1 = [
                                [la-lh,  0,override_be,  state_map['i']],
                                [la-lh,  0,override_be+1,state_map['i']],
                                [la-lh-1,1,override_be,  state_map['r']],
                                [la-lh-1,0,override_be,  state_map['i']]
                            ]
                            prob_1 = prob_0
                            reward_1 = make_reward(la,lh,be,s,1)
                            append_in_list(1,l,s,result_1,prob_1,reward_1)
                            # row2 end
                        else:
                            set_unable(1)

                            # row4
                        if la >= lh:
                            if la != 0:
                                result_match_a_wait = [
                                    [la+1,lh,be,  state_map['a']],
                                    [la+1,lh,be+1,state_map['a']],
                                    [la-lh,1,be-math.ceil(lh*be/la),  state_map['r']],
                                    [la,lh+1,be,  state_map['r']],
                                    [la,lh,  be,  state_map['a']]
                                ]
                            else:
                                result_match_a_wait = [
                                    [la+1,lh,be,  state_map['a']],
                                    [la+1,lh,be+1,state_map['a']],
                                    [0 ,1, 0,  state_map['r']],
                                    [la,lh+1,be,  state_map['r']],
                                    [la,lh,  be,  state_map['a']]
                                ]

                            prob_match_a_wait = [self.alpha,
                                                 self.omega,
                                                 self.gamma*(1-self.alpha-self.omega)*(1-self.rs),
                                                 (1-self.gamma)*(1-self.alpha-self.omega)*(1-self.rs),
                                                 (1-self.alpha-self.omega)*self.rs]
                            reward_match_a_wait = make_reward(la,lh,be,s,3)
                            # index 2 for wait
                            if s == state_map['a']:  # col 4
                                if cut:
                                    set_unable(2)
                                else:
                                    append_in_list(2,l,s,result_match_a_wait,prob_match_a_wait,reward_match_a_wait)

                            # index 3 for match also la >= lh
                            if s == state_map['r'] and not cut:
                                append_in_list(3,l,s,result_match_a_wait,prob_match_a_wait,reward_match_a_wait)
                            else:
                                set_unable(3)
                            # match action end
                        else:
                            set_unable(3)
                            if s == state_map['a']:
                                set_unable(2)
                            # s = i or r col 3
                        if s != state_map['a'] :  # col 4
                            if cut:
                                set_unable(2)
                            else:
                                result_wait = [
                                    [la+1,lh,be,state_map['i']],
                                    [la+1,lh,be+1,state_map['i']],
                                    [la,lh+1,be,state_map['r']],
                                    [la,lh,be,state_map['i']]
                                ]
                                prob_wait = prob_0
                                reward_wait = make_reward(la,lh,be,s,2)
                                append_in_list(2,l,s,result_wait,prob_wait,reward_wait)
                        # row 3 end
                        if with_exit:
                            # how to express exit?????
                            if la > lh and la > self.k:
                                append_in_list(4,l,s,[exit_state],[1],[la-be+self.vd])  # trans to exit state
                            else :
                                set_unable(4)
        action_num = 4
        if with_exit:
            action_num = 5
            append_in_list(4,exit_state[0:3],exit_state[3],[exit_state],[1],[0])
            append_in_list(3,exit_state[0:3],exit_state[3],[exit_state],[1],[-10**9])
            append_in_list(2,exit_state[0:3],exit_state[3],[exit_state],[1],[-10**9])
            append_in_list(1,exit_state[0:3],exit_state[3],[exit_state],[1],[-10**9])
            append_in_list(0,exit_state[0:3],exit_state[3],[exit_state],[1],[-10**9])
            P = [csr_matrix((P_probability[i],(pre_state[i],result_state[i])),
                            shape=(self.cutoff**3*len(state_space) +1,self.cutoff**3*len(state_space) +1))
                for i in range(action_num)]

            R = [csr_matrix((R_reward[i],(pre_state[i],result_state[i])),
                            shape=(self.cutoff**3*len(state_space) +1,self.cutoff**3*len(state_space) +1))
                for i in range(action_num)]
        else:
            P = [csr_matrix((P_probability[i],(pre_state[i],result_state[i])),
                            shape=(self.cutoff**3*len(state_space),self.cutoff**3*len(state_space)))
                for i in range(action_num)]

            R = [csr_matrix((R_reward[i],(pre_state[i],result_state[i])),
                            shape=(self.cutoff**3*len(state_space),self.cutoff**3*len(state_space)))
                for i in range(action_num)]

        return P, R

