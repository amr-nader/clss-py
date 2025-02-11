import numpy as np
import casadi as cs
from robots.quadrotor_indi import computeControl, computeTheta, \
    computeDynamics,computePIDynamic, computeInputTube, computeStateTube

import matplotlib.pyplot as plt
from utils import traj_gen as traj

N_rollout = 200
Tsim = 2.5
dt =  1/50
Nsim = Tsim/dt

# Reference setpoint
p_initial = np.array([0,0,0])
p_final = np.array([0.5,0,0.5])
total_time = 2

# Reference setpoint
#x_r = np.array([0.05,0,0.05,np.deg2rad(0),np.deg2rad(0),np.deg2rad(0),0,0,0,0,0,0,0,0,0,0,0,0])

# initial state
nx = 12
x0 = np.array([0,0,0,np.deg2rad(0),np.deg2rad(0),np.deg2rad(0),0,0,0,0,0,0])


# hexarotor parameters
npar = 9

m_v = 0.5
I_v_x = 0.1
I_v_y = 0.1
I_v_z = 0.1
r_c_x = 0
r_c_y = 0
r_c_z = 0
k_f = 0.000067
k_tau =  0.000001
par = np.array([m_v,I_v_x,I_v_y,I_v_z,r_c_x,r_c_y,r_c_z,k_f,k_tau])
del_p = np.array([m_v*0.1,I_v_x*0.1,I_v_y*0.1,I_v_z*0.1,r_c_x+0.001,r_c_y+0.001,r_c_z-0.003,k_f*0.1,k_tau*0.1])

kp = 5
kd = 2*np.sqrt(kp)
kR = 150
kom = 2*np.sqrt(kR)

k = [kp,kp,kp,kd,kd,kd,kR,kR,kR,kom,kom,kom]

x = x0


PI_0 = np.zeros((nx,npar))
PI = PI_0
state_history = []
stateTube_history = []
input_history = []
inputTube_history = []

plt.figure(figsize=(10, 5))

for i in range(N_rollout):

    state_history = []
    stateTube_history = []
    input_history = []
    input_Tubehistory = []

    a_f = np.array([0,0,-9.81,0,0,0])
    u_f = np.array([0,0,0,0])
    dafdp = np.zeros((6,npar))
    dufdp = np.zeros((4,npar))
    x = x0
    PI = PI_0


    if i != 0:
        val = del_p  # Replace with your desired value
        random_value = np.random.uniform(-val, val)
        par_n = par + random_value
    else:
        par_n = par
    
    for j in range(int(Nsim)):
        curr_t = j*dt
        q_outx,qdt_outx,qddt_outx = traj.computeTrajectory(0,0.5,curr_t,total_time)
        q_outy,qdt_outy,qddt_outy = traj.computeTrajectory(0,0,curr_t,total_time)
        q_outz,qdt_outz,qddt_outz = traj.computeTrajectory(0,0.5,curr_t,total_time)

        q_out = np.array([q_outx,q_outy,q_outz])
        qdt_out = np.array([qdt_outx,qdt_outy,qdt_outz])
        qddt_out = np.array([qddt_outx,qddt_outy,qddt_outz])
        #x_r = np.array([q_out,qdt_out,qddt_out])
        x_r = np.concatenate([q_out,[0, 0, 0], qdt_out, [0, 0, 0], qddt_out, [0, 0, 0]])
        #x_r = np.array([q_outx,0,0,0,0,0,qdt_outx,0,0,0,0,0,qddt_outx,0,0,0,0,0])

        
        if i==0:
            u = computeControl(x,x_r,par,k,a_f,u_f)
            x_dot = computeDynamics(x,u,par)
            PI_dot = computePIDynamic(x,u,par,x_r,k,a_f,u_f,dafdp,dufdp,PI)
            theta = computeTheta(x,u,par,x_r,k,a_f,u_f,dafdp,dufdp,PI)
            del_x = computeStateTube(PI,del_p)
            del_u = computeInputTube(x,u,par,x_r,k,a_f,u_f,dafdp,dufdp,PI,del_p)

            
            PI = PI + PI_dot*dt
            x = x + x_dot*dt
            a_f = x_dot[6:]
            u_f = u
            dafdp = PI_dot[6:,:] 
            dufdp = theta

            state_history.append({"step": j, "state": x.full().tolist()})
            stateTube_history.append({"step": j, "state": del_x.full().tolist()})
            input_history.append({"step":j, "state":u.full().tolist()})
            input_Tubehistory.append({"step":j, "state":del_u.full().tolist()})
        else:
            u = computeControl(x,x_r,par,k,a_f,u_f)
            x_dot = computeDynamics(x,u,par_n)
            
            x = x + x_dot*dt
            a_f = x_dot[6:]
            u_f = u
            
            state_history.append({"step": j, "state": x.full().tolist()})
            input_history.append({"step":j, "state":u.full().tolist()})

    
    #cmd = [state[0][0] for state in u]
    #del_cmd = [state[0][0] for state in del_u]

    time_steps = [record["step"] for record in state_history]
    x = [record["state"] for record in state_history]
    del_x = [record["state"] for record in stateTube_history]
    u = [record["state"] for record in input_history]
    del_u = [record["state"] for record in input_Tubehistory]


    x_pos = [state[0][0] for state in x]
    y_pos = [state[1][0] for state in x]
    z_pos = [state[2][0] for state in x]

    del_x_pos = [state[0][0] for state in del_x]
    del_y_pos = [state[1][0] for state in del_x]
    del_z_pos = [state[2][0] for state in del_x]

    roll = [state[3][0] for state in x]
    pitch = [state[4][0] for state in x]
    yaw = [state[5][0] for state in x]

    del_roll = [state[3][0] for state in del_x]
    del_pitch = [state[4][0] for state in del_x]
    del_yaw = [state[5][0] for state in del_x]

    if i==0:
        plt.subplot(3,1,1)
        plt.plot(time_steps,np.array(x_pos) + np.array(del_x_pos), marker="o",color='k',linewidth=3)
        plt.plot(time_steps, x_pos, label="Position", marker=".",color='r',linewidth=2)
        plt.plot(time_steps, np.array(x_pos) - np.array(del_x_pos), marker="o",color='k',linewidth=3)
        plt.grid(True) 
        plt.subplot(3,1,2)
        plt.plot(time_steps,np.array(z_pos) + np.array(del_z_pos), marker="o",color='k',linewidth=3)
        plt.plot(time_steps, z_pos, label="Position", marker=".",color='r',linewidth=2)
        plt.plot(time_steps, np.array(z_pos) - np.array(del_z_pos), marker="o",color='k',linewidth=3)
        plt.grid(True)
        plt.subplot(3,1,3)
        plt.plot(time_steps,np.array(pitch) + np.array(del_pitch), marker="o",color='k',linewidth=3)
        plt.plot(time_steps, pitch, label="Position", marker=".",color='r',linewidth=2)
        plt.plot(time_steps, np.array(pitch) - np.array(del_pitch), marker="o",color='k',linewidth=3)
        plt.grid(True)
    else:
        plt.subplot(3,1,1)
        plt.plot(time_steps, x_pos, label="Position", marker=".",linewidth=0.05)
        plt.subplot(3,1,2)
        plt.plot(time_steps, z_pos, label="Position", marker=".",linewidth=0.05)
        plt.subplot(3,1,3)
        plt.plot(time_steps, pitch, label="Position", marker=".",linewidth=0.05)

# Show plots
plt.tight_layout()
plt.show()





