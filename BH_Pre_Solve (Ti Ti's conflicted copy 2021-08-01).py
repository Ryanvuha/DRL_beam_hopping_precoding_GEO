# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:50:50 2021

@author: vu-nguyen.ha
"""
import cvxpy as cp
import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
#import scipy.linalg as la
#import cvxpy.atoms.elementwise.abs as cpabs
import cmath


#### Loading Channel .Mat File ------------------------------------------------
def Channel_Load(file_name):
    temp01 = sio.loadmat(file_name)
    H_channel = temp01['ChMat']
    
    return H_channel

#### Updating Compress Sensing Weights ----------------------------------------
def Weight_Update(Power_Vector,epsilon):
    Psi = np.sqrt(1/(pow(Power_Vector,2)+epsilon))
    
    return Psi

#### SDP-based CVX Solution 
# H_tslot = np.asmatrix(H_channel[index_slted-users,:,time_slot])
def SDP_Solution(H_tslot,Psi,P_beam,g,K_T,rho,sigma):
    
    num_beam = H_tslot.shape[1]
    num_user = H_tslot.shape[0]
    
    
    ## Variable creating
    W = {}
    Z = {}
    
    constraints = []
    
    for uu in range(num_user):
        W[uu] = cp.Variable((num_beam,num_beam), symmetric = True) # symetric definition
        Z[uu] = cp.Variable((num_beam,num_beam))
        #X[uu] = cp.Variable((num_beam,num_beam), complex = True) # complex definition
        constraints += [W[uu]+cmath.sqrt(-1)*Z[uu] >> 0]
        constraints += [Z[uu].T + Z[uu] == 0]
        #constraints += [W[uu] == X[uu]] 
    
    # E matrix for SDP problem
    #E=np.zeros((num_beam,num_beam))
    #for bb in range(num_beam):
    #    E[bb,bb] = 1+rho*Psi[bb]
        
    E = np.diag(1+rho*Psi[0],k=0)
    
    Obj1 = 0
    
    
    for uu in range(num_user):
        Obj1 += cp.norm(cp.trace(E @ W[uu]))
        Temp1 = H_tslot[uu,:].getH()*H_tslot[uu,:]
        Tem1_re = np.real(Temp1)
        Tem1_im = np.imag(Temp1)
        #for bb in range(num_beam):
        #    Temp_1[bb,bb] = np.abs(Temp_1[bb,bb])
        SINR_temp = cp.trace(Tem1_re @ W[uu] - Tem1_im @ Z[uu])
        for jj in range(num_user):
            if jj!=uu:
                SINR_temp += -g[uu]*cp.trace(Tem1_re @ W[jj] - Tem1_im @ Z[jj])

        constraints += [SINR_temp >= g[uu]*sigma]
    
    K_temp = 0    
    for bb in range(num_beam):
        P_temp = 0
        for uu in range(num_user):
            P_temp += W[uu][bb,bb]
            
        constraints += [P_temp <= P_beam]
        K_temp += Psi[0][bb]*P_temp
       
    constraints += [K_temp <= K_T]
    
    
    obj = cp.Minimize(Obj1)   
    prob = cp.Problem(obj, constraints)
    #prob.solve(solver=cp.MOSEK)
    prob.solve()        

    

    print("Optimal status: %s" % prob.status)

    if prob.status not in ["infeasible", "unbounded"]:
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)


    return [W,Z]

##############################################################################
########## Wei Yu's Uplink-Downlink Duality Method
##############################################################################

### Lambda Update ###################################

def Update_lambda(H_tslot,Psi,Q,g,rho,alpha,esp):
    
    num_beam = H_tslot.shape[1]
    num_user = H_tslot.shape[0]
    
    E = np.diag(1+rho*Psi[0],k=0)
    Q_matrix = np.diag(Q,k=0)
    Psi_matrix =  np.diag(Psi[0],k=0)
    
    lambda_0 = np.zeros((num_user,1))
    lambda_1 = np.ones((num_user,1))
    
    num_ite = 0
    
    while (np.linalg.norm(lambda_1-lambda_0) > esp) and (num_ite < 10000):
        num_ite +=1
        
        #if (num_ite%2 == 0): 
        #print('--------- lambda loop iteration '+ str(num_ite)+' gap '+ str(np.linalg.norm(lambda_1-lambda_0)))
        
        lambda_0 = np.copy(lambda_1)
        
        sum_temp = np.zeros((num_beam,num_beam),dtype = 'complex_')
        
        sum_temp += E+Q_matrix+alpha*Psi_matrix
        
        for jj in range(num_user):
            temp1 = lambda_1[jj][0]*(H_tslot[jj,:].getH()*H_tslot[jj,:])
            #sum_temp += lambda_1[jj][0]*(H_tslot[jj,:].getH()*H_tslot[jj,:])
            sum_temp = np.add(sum_temp, temp1, out=sum_temp, casting="unsafe")
        
        #if np.linalg.cond(sum_temp) < 1/np.finfo(sum_temp.dtype).eps:
        sum_inv = np.linalg.inv(sum_temp)
        #else:
        #    sum_inv = np.linalg.pinv(sum_temp)
        
        for uu in range(num_user):
            temp_1 = (1+(1/g[uu]))*H_tslot[uu,:]*sum_inv*H_tslot[uu,:].getH()
            temp_2 = (temp_1)**(-1)
            lambda_1[uu] = np.real(temp_2)
            
    return lambda_1
            
### Precoding Update ################################

def Update_w(H_tslot,Psi,Q,g,rho,sigma,alpha,lamb):
    
    num_beam = H_tslot.shape[1]
    num_user = H_tslot.shape[0]
    
    E = np.diag(1+rho*Psi[0],k=0)
    Q_matrix = np.diag(Q,k=0)
    Psi_matrix =  np.diag(Psi[0],k=0)
    
    sum_temp = np.zeros((num_beam,num_beam),dtype = 'complex_')
        
    sum_temp += E+Q_matrix+alpha*Psi_matrix
        
    for jj in range(num_user):
        temp1 = lamb[jj][0]*(H_tslot[jj,:].getH()*H_tslot[jj,:])
        #sum_temp += lambda_1[jj][0]*(H_tslot[jj,:].getH()*H_tslot[jj,:])
        sum_temp = np.add(sum_temp, temp1, out=sum_temp, casting="unsafe")
        
    sum_inv = np.linalg.inv(sum_temp)
   # print(str(sum_inv.shape))
    
    
    W_hat = np.asmatrix(np.zeros((num_beam,num_user),dtype = 'complex_'))
    G_temp = np.zeros((num_user,num_user))
    for uu in range(num_user):
        W_hat[:,uu] = sum_inv*H_tslot[uu,:].getH()
        
        for jj in range(num_user):
            G_temp[jj,uu] = -np.abs(H_tslot[jj,:]*W_hat[:,uu])**2
            if jj==uu:
                G_temp[jj,uu] = -G_temp[jj,uu]/g[jj]
    
    delta = sigma*np.linalg.inv(G_temp)*np.asmatrix(np.ones((num_user,1)))
    
   # print(str(delta))
    
    delta_matrix = np.zeros((num_user,num_user),dtype = 'complex_')
    
    for uu in range(num_user):
        delta_matrix[uu,uu] = delta[uu]
    
    #print(str(delta_matrix.shape))
    
    #W = np.asmatrix(np.zeros((num_beam,num_user)))
    
    #for uu in range(num_user):
    #    W[:,uu] = np.sqrt(delta[uu][0])*W_hat[:,uu]
    
    W = W_hat*np.sqrt(delta_matrix)
    
    return W


#### Uplink-Downlink Duality Solution

def UD_Dual_Solve(H_tslot,Psi,P_beam,g,K_T,rho,sigma,Q,alpha,st_size,eps1,esp2):
    
    num_beam = H_tslot.shape[1]
    #num_user = H_tslot.shape[0]
    
    Obj_0 = 0
    Obj_1 = 1
    
    Pt = np.zeros((num_beam,1)) 
    num_ite = 0
    
    while np.abs(Obj_1-Obj_0) > eps1:
        num_ite +=1
        
        print('iteration '+ str(num_ite)+' gap '+ str(np.linalg.norm(Obj_1-Obj_0)))
        
        Obj_0 = np.copy(Obj_1)
        
        # step 1: calculate lambda
        lamb = Update_lambda(H_tslot,Psi,Q,g,rho,alpha,esp2)
    
        # step 2: calculate precoding
        W = Update_w(H_tslot,Psi,Q,g,rho,sigma,alpha,lamb)
        
        for nn in range(num_beam):
            temp1 = W[nn,:]*W[nn,:].getH()
            temp2 = temp1.item()
            Pt[nn] = np.real(temp2)
            
        P_sum = 0
        Obj_1 = 0
        
        for nn in range(num_beam):
            if (Pt[nn]-P_beam) < 0:
                st_size[nn] = st_size[nn]/1.5;
            
            st_size[nn] = st_size[nn]/1.05;
            Q[nn] += st_size[nn]*(Pt[nn]-P_beam)
            if Q[nn] < 0:
                Q[nn] = 0
                
            P_sum += Psi[0][nn]*Pt[nn]
            #P_sum += Psi[nn]*Pt[nn]
            Obj_1 += (1+rho*Psi[0][nn])*Pt[nn]
            #Obj_1 += (1+rho*Psi[nn])*Pt[nn]
            
        if (P_sum - K_T) < 0:
            st_size[-1] = st_size[-1]/1.5;
        
        st_size[-1] = st_size[-1]/1.05;
        alpha += st_size[-1]*(P_sum - K_T)
        
        
        
    return [W,Pt,Obj_1,Q,alpha,st_size]
            
        
def SINR_cal(H_tslot,W,sigma):
    
    num_user = H_tslot.shape[0]
    SINR = np.zeros((num_user,1))
    
    for uu in range(num_user):
        
        temp1 = np.copy(sigma)
        for jj in range(num_user):
            temp1 += np.asscalar(np.abs(H_tslot[uu,:]*W[:,jj])**2)
            #np.add(temp1, temp1a, out=temp1, casting="unsafe")
            
        temp2 = np.abs(H_tslot[uu,:]*W[:,uu])**2
        temp3 = temp2/(temp1-temp2)
        SINR[uu] = np.real(temp3)
        
    return SINR


def CS_Solution(H_tslot,P_beam,g,K_T,rho,sigma,Q,alpha,st_size,eps1,eps2,eps3,eps4):
    
    num_beam = H_tslot.shape[1]
    
    Psi_0 =10*np.abs(np.random.randn(1,num_beam))
    Psi_1 =10*np.abs(np.random.randn(1,num_beam))
    
    num_ite = 0
    
    while (np.linalg.norm(Psi_1-Psi_0) > eps3) and (num_ite < 100):
        
        num_ite += 1
        
        [W,Pt,Obj_1,Q,alpha,st_size] = UD_Dual_Solve(H_tslot,Psi_1,P_beam,g,K_T,rho,sigma,Q,alpha,st_size,eps1,eps2)
        
        Psi_0 = np.copy(Psi_1)
        
        for nn in range(num_beam):
            Psi_1[0][nn] = np.sqrt(1/(Pt[nn]**2+eps4))
            
    return [W,Pt,Obj_1]    
    
        
    
    
    

# Ti write -------------------------------------------------------------------------- 
    
    
    
    
def mapping_MODCOD(scheme_l):
    # Table 20a - DVB Document A082-2 Rev.2
    SNR_array_dB = np.array([-2.85, -2.03, 0.22, 1.45, 4.73, 5.13, 6.12, 7.02, 7.49, 5.97, 6.55, 6.84, 7.51, 7.8, 7.41, 
                          8.10, 8.38,8.43, 9.27, 9.71, 10.65, 11.99, 11.10, 11.75, 12.17, 13.05, 13.98, 14.81, 15.47,
                          15.87, 16.55,17.73, 18.53, 16.98, 17.24, 18.10, 18.59, 18.84, 19.57])
    
    SNR_array = 10**(SNR_array_dB/10)
    rate_array = np.array([0.434841, 0.567805, 0.889135, 1.088581, 1.647211, 1.713601, 1.896173, 2.062148, 2.145136, 1.972253,
                           2.104850, 2.193247, 2.281645,  2.370043, 20370043, 2.458441, 2.524739, 2.635236, 2.745734, 2.856231,
                           3.077225, 3.386618, 3.291954, 3.510192, 3.620536, 3.841226, 4.206428, 4.338659, 4.603122, 4.735354, 4.936639, 
                           5.1663248, 5.355556, 5.065690, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
    
    R = rate_array[scheme_l]
    g = SNR_array[scheme_l]
    
    return R, g
    
    
        
########## Running -----------------------------------------------------------
# H_Ch = Channel_Load('ML_Channel_Matrix_Fixed_Pos_10Users_49Beams_50Time-Slot1th_Realizations_over_3000.mat')
# H_tslot = np.asmatrix(H_Ch[:,:,1])

def get_state_chan(H_tslot):
    ## for each user, record min, mean, and max of abs(H[k])
    
    Habs = abs(H_tslot)
    
    # Hmin = np.min(Habs, axis=1)
    # Hmean = np.mean(Habs, axis=1)
    # Hmax = np.max(Habs, axis=1)
    
    # state_chan = np.asarray(Hmin).flatten()
    # state_chan = np.append(state_chan, np.asarray(Hmean).flatten())
    # state_chan = np.append(state_chan, np.asarray(Hmax).flatten())
    
    state_chan = np.array([])
    for ii in range(Habs.shape[0]):
        qq = np.sort( np.asarray(Habs[ii]).flatten())
        state_chan = np.append(state_chan, qq[-3:])
    
    return state_chan


def get_state(H_tslot, remain_Q, remain_T):
    state_chan = get_state_chan(H_tslot)
    
    state = np.append(state_chan, remain_Q)
    state = np.append(state, remain_T)
    
    return state



def get_reward(H_tslot, action_round, remain_Q, remain_T , Obj_max):
    
    
    R , g = mapping_MODCOD(action_round)
    num_beam = H_tslot.shape[1]
    num_user = H_tslot.shape[0]
    
    np.random.seed(1)
    #Psi =10*np.abs(np.random.randn(1,num_beam))
    #g = [1,2,3,4,5,6,7,8,9,10]
    P_beam = 5 
    K_T = 10
    rho = 0.1
    sigma=10e-14
    
    
    
    Q=np.ones((num_beam,1))*2
    alpha = 2
    st_size = np.ones((num_beam+1,1))
    eps1 = 1e-3
    eps2 = 0.01 
    eps3 = 1e-4 # 1e-8
    eps4 = 1e-4 #1e-8
    
    [W,Pt,Obj_1] = CS_Solution(H_tslot,P_beam,g,K_T,rho,sigma,Q,alpha,st_size,eps1,eps2,eps3,eps4)
    
    SINR = SINR_cal(H_tslot,W,sigma)
    
    
    ##################
    
    Pt1 = np.asarray(Pt).flatten()
    Pt2 = np.multiply(Pt1, Pt1> np.max(Pt1)/500)
    
    cons_10c = np.where(np.asarray(SINR).flatten() - g < -0.01)
    cons_10d = np.linalg.norm(Pt2,0)-K_T > 0
    
    for k in range(num_user):        
        remain_T[k] = np.max((remain_T[k] - 1, 0 ));
        if remain_T[k] >0 :
            remain_Q[k] = remain_Q[k] - R[k]
        
    flag_data_check = False
    for k in range(num_user):
        if (remain_T[k] ==0) & (remain_Q[k] > 0):
            flag_data_check = True
            break
        
    
    if flag_data_check:
        reward = 0
    else:
        if (cons_10c[0].shape[0] > 0) | cons_10d :
            reward = 0
        else:
            reward = np.max((0,Obj_max - Obj_1[0]))
            
    
    return reward, remain_Q, remain_T




from DDPG_ref import OUActionNoise, get_actor, get_critic, policy, Buffer, update_target
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from DDPG_ref import OUActionNoise, get_actor, get_critic, policy, Buffer, update_target
import matplotlib.pyplot as plt


    
    
    
std_dev = 0.1
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


string1 = './10beams_3Users_20TimeSlot/ML_Channel_Matrix_Fixed_Pos_3Users_10Beams_20Time-Slot'
string2 = 'th_Realizations_over_3000.mat'

strings = string1+str(1)+string2
    
H_Ch = Channel_Load(strings)
    
T = H_Ch.shape[2]
    
# Takes about 4 min to train

num_user = 3
num_beam = 10

num_states = 5*num_user 
num_actions = num_user
num_code = 39 # from Table 20.a

prev_state = np.random.rand(num_states)








remain_T0 = np.random.randint(int(2*T/3),T,num_user)
remain_Q0 = np.multiply(remain_T0, 2 + np.random.rand(num_user))
Obj_max = 500   
    
for ep in range(1, total_episodes):
    
    remain_T = np.copy(remain_T0)
    remain_Q = np.copy(remain_Q0)

    strings = string1+str(ep)+string2
    
    H_Ch = Channel_Load(strings)
    
    T = H_Ch.shape[2]
    
    done = False
    episodic_reward = 0
    for t in range(T):
        if t==T:
            done = True
            
        H_tslot = np.asmatrix(H_Ch[:,:,t])
            

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise, actor_model)

        action_round = np.round((num_code-1)*action).astype(int)
        
        # Recieve state and reward from environment.
        state = get_state(H_tslot, remain_Q, remain_T)
        reward, remain_Q, remain_T = get_reward(H_tslot, action_round, remain_Q, remain_T , Obj_max)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn(target_actor, target_critic, critic_model, actor_model, critic_optimizer, actor_optimizer, gamma)
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)


        prev_state = state
        

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()










