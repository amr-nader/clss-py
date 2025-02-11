import casadi as ca
import numpy as np
from utils import helper_utils as util
import math

def get_alloc_gtmr(al,be,c_f,c_t,l_arm):
    yaw_i = 0
    k_i = 1 #sign of first propeller -1 = cw
    b_p_p1 = ca.vertcat(l_arm,0,0)
    W_map = ca.SX.zeros(6,6)

    for i in range(6):

        # The propeller pose in frame b
        b_R_pi = util.Rz(yaw_i)@util.Rx(al[i])@util.Ry(be[i])

        b_z_pi = b_R_pi[:,2]
        b_p_pi = util.Rz(yaw_i)@b_p_p1

        # Drag and Thrust of prop_i expressed in Frame b
        temp = util.cross_prod(b_p_pi,b_z_pi)
        b_m_i = k_i*c_t @ (b_z_pi) + c_f @ temp
        b_f_i = c_f*(b_z_pi)
        W_map[0:3,i] = b_f_i
        W_map[3:6,i] = b_m_i

        yaw_i = yaw_i + (math.pi/3)
        k_i = k_i*-1

    return W_map

# hexarotor state 
nx = 12
x = ca.SX.sym('x',nx,1)
p_v = x[0:3]
phi_v = x[3:6]
R_v = util.rpy2Rot(phi_v)
v_v = x[6:9]
om_v = x[9:12]
phi_dot  = util.rpyDerivative(om_v,phi_v)

# hexarotor inputs - square of propeller speeds
nu = 6
u = ca.SX.sym('u',nu,1) 

# hexarotor parameters
npar = 10
p = ca.SX.sym('p',npar,1)
m_v = p[0]
I_v = p[1:4]
MoI = ca.diag(I_v)
r_c = p[4:7]
k_f = p[7]
k_tau = p[8]
alpha = p[9]


# hexarotor dynamics 

# thrust model
beta = 0.0
al = ca.vertcat(alpha,-alpha,alpha,-alpha,alpha,-alpha)
be = ca.vertcat(-beta,beta,-beta,beta,-beta,beta)
l_arm = 0.1550
F = util.get_alloc_gtmr(al,be,k_f,k_tau,l_arm)


# Equations of Motion
# M x_ddot + h(x,x_dot) = F(x)u
M = ca.vertcat(ca.horzcat(m_v @ ca.SX.eye(3), -m_v @ util.skewMat(r_c)),
               ca.horzcat(m_v @ util.skewMat(r_c), MoI - m_v @ util.skewMat(r_c) @ util.skewMat(r_c)))

h = ca.vertcat(m_v*9.81*R_v.T @ ca.vertcat(0.0,0.0,1.0) \
           + m_v@util.skewMat(om_v)@util.skewMat(om_v)@r_c,
           m_v*9.81*util.skewMat(r_c)@R_v.T @ ca.vertcat(0.0,0.0,1.0) \
            + util.skewMat(om_v)@(MoI - util.skewMat(r_c)*util.skewMat(r_c))@om_v) 

# sum of forces and torques
f_tot = F[0:3,:]@u - h[0:3]
tau_tot = F[3:6,:]@u - h[3:6]

dyn =  ca.solve(M,ca.vertcat(f_tot,tau_tot))

# state space model
x_dot = ca.vertcat(v_v,phi_dot,R_v@dyn[0:3],dyn[3:6])
computeDynamics = ca.Function('computeDynamics',[x,u,p],[x_dot])

# Hexarotor Controller
nk = 12
k = ca.SX.sym('k',nk,1)
kp = ca.diag(k[0:3])
kd = ca.diag(k[3:6])
kR = ca.diag(k[6:9])
kom = ca.diag(k[9:12])

nref = 18
x_r = ca.SX.sym('x_r',nref,1)
R_r = util.rpy2Rot(x_r[3:6])


virt_cmd = ca.vertcat(x_r[12:15] + kp@(x_r[0:3]-p_v) + kd@(x_r[6:9]-v_v),
                      x_r[15:18] + kR@util.eSO3(R_r,R_v) + kom@(x_r[9:12]-om_v))
                      
w_cmd = M@virt_cmd + h
u_cmd = ca.solve(ca.vertcat(R_v@F[0:3,:],F[3:6,:]),w_cmd)
computeControl = ca.Function('computeControl',[x,x_r,p,k],[u_cmd])

# Compute Sensitivites 
dfdx = ca.jacobian(x_dot,x)
dfdu = ca.jacobian(x_dot,u)
dfdp = ca.jacobian(x_dot,p)
dhdq = ca.jacobian(u_cmd,x)

PI = ca.SX.sym('PI',nx,npar)

PI_dot = (dfdx+dfdu@dhdq)@PI + dfdp
Theta = dhdq@PI

computePIDynamic = ca.Function('computePIDynamic',[x,u,p,x_r,k,PI],[PI_dot])
computeTheta = ca.Function('computeTheta',[x,u,p,x_r,k,PI],[Theta])

# ComputeTubes
del_x = ca.SX.sym('del_x',nx,1)
del_u = ca.SX.sym('del_u',nu,1)
del_p = ca.SX.sym('del_p',npar,1)
W_par = ca.diag(del_p**2)

for i in range(nx):
    J = PI[i,:]@W_par@PI[i,:].T
    del_x[i] = ca.sqrt(J)

computeStateTube = ca.Function('computeStateTube',[PI,del_p],[del_x])

for i in range(nu):
    J = Theta[i,:]@W_par@Theta[i,:].T
    del_u[i] = ca.sqrt(J)

computeInputTube = ca.Function('computeInputTube',[x,u,p,x_r,k,PI,del_p],[del_u])
