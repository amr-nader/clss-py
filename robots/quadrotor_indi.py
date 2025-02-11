import casadi as ca
import numpy as np
from utils import helper_utils as util



def get_quad_alloc(c_f,c_t,l_arm):
    W_map = ca.SX.zeros(6,4)

    W_map[2,:] = ca.horzcat(c_f,c_f,c_f,c_f)
    W_map[3,1] = c_f*l_arm
    W_map[3,3] = -c_f*l_arm
    W_map[4,0] = -c_f*l_arm
    W_map[4,2] = c_f*l_arm
    W_map[5,:] = ca.horzcat(c_t,-c_t,c_t,-c_t)

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
nu = 4
u = ca.SX.sym('u',nu,1) 

# hexarotor parameters
npar = 9
p = ca.SX.sym('p',npar,1)
m_v = p[0]
I_v = p[1:4]
MoI = ca.diag(I_v)
r_c = p[4:7]
k_f = p[7]
k_tau = p[8]


# hexarotor dynamics 

# thrust model
l_arm = 0.1550
F = get_quad_alloc(k_f,k_tau,l_arm)


# Equations of Motion
# M x_ddot + h(x,x_dot) = F(x)u
M = ca.vertcat(ca.horzcat(m_v @ ca.SX.eye(3), -m_v @ util.skewMat(r_c)),
               ca.horzcat(m_v @ util.skewMat(r_c), MoI - m_v @ util.skewMat(r_c) @ util.skewMat(r_c)))

h = ca.vertcat(m_v*9.81*R_v.T @ ca.vertcat(0.0,0.0,1.0) \
           + m_v@util.skewMat(om_v)@util.skewMat(om_v)@r_c,
           m_v*9.81*util.skewMat(r_c)@R_v.T @ ca.vertcat(0.0,0.0,1.0) \
            + util.skewMat(om_v)@(MoI - util.skewMat(r_c)*util.skewMat(r_c))@om_v) 

# sum of forces and torques
#f_tot = R_v.T@F[0:3,:]@u - h[0:3]
f_tot = F[0:3,:]@u - h[0:3]
tau_tot = F[3:6,:]@u - h[3:6]

dyn =  ca.solve(M,ca.vertcat(f_tot,tau_tot))

# state space model
x_dot = ca.vertcat(v_v,phi_dot,R_v@dyn[0:3],dyn[3:6])
computeDynamics = ca.Function('computeDynamics',[x,u,p],[x_dot])

# quadrotor INDI Controller

F_bar = ca.SX.zeros(4,4)
F_bar = F[2:,:]

a_f = ca.SX.sym('a_f',3,1)
om_f_dot = ca.SX.sym('om_f_dot',3,1)
xdotf = ca.vertcat(a_f,om_f_dot)


u_f = ca.SX.sym('u_f',4,1)

w_f = F@u_f #wrench in body frame
f_f = R_v@w_f[:3]
m_f = w_f[3:]


nk = 12
k = ca.SX.sym('k',nk,1)
kp = ca.diag(k[0:3])
kd = ca.diag(k[3:6])
kR = ca.diag(k[6:9])
kom = ca.diag(k[9:12])

nref = 18
x_r = ca.SX.sym('x_r',nref,1)
R_r = util.rpy2Rot(x_r[3:6])

v_virt = x_r[12:15] + kp@(x_r[0:3]-p_v) + kd@(x_r[6:9]-v_v) 
f_des = m_v*(v_virt - a_f) + f_f # f_des is in world frame
f_cmnd = f_des.T@(R_v@ca.vertcat(0,0,1))

z_b_des = f_des/ca.norm_2(f_des)
x_bp_des = util.Rz(x_r[5])@ca.vertcat(1,0,0)
y_b_des = util.cross_prod(z_b_des,x_bp_des)/(ca.norm_2(util.cross_prod(z_b_des,x_bp_des)))
x_b_des  = util.cross_prod(y_b_des,z_b_des)
#R_v_ref = ca.SX.eye(3)
R_v_ref = ca.horzcat(x_b_des, y_b_des, z_b_des)

om_dot_virt = kR@util.eSO3(R_v_ref,R_v) - kom@om_v 
m_cmnd = MoI@(om_dot_virt - om_f_dot) + m_f
#- I_v@(util.skewMat(om_v)@R_v.T@R_v_ref*om_v_ref)

cmnd = ca.vertcat(f_cmnd,m_cmnd)


u_cmd = ca.solve(F_bar,cmnd)
computeControl = ca.Function('computeControl',[x,x_r,p,k,xdotf,u_f],[u_cmd])


# Compute Sensitivites 
dfdx = ca.jacobian(x_dot,x)
dfdu = ca.jacobian(x_dot,u)
dfdp = ca.jacobian(x_dot,p)
dhdx = ca.jacobian(u_cmd,x)

dhdxdotf = ca.jacobian(u_cmd,xdotf)

dhduf = ca.jacobian(u_cmd,u_f)

dafdp = ca.SX.sym('dafdp',6,npar)
dufdp = ca.SX.sym('dufdp',4,npar)

PI = ca.SX.sym('PI',nx,npar)

PI_dot = (dfdx+dfdu@dhdx)@PI + dfdu@(dhdxdotf@dafdp+dhduf@dufdp) + dfdp
Theta = dhdx@PI + dhdxdotf@dafdp + dhduf@dufdp

computePIDynamic = ca.Function('computePIDynamic',[x,u,p,x_r,k,xdotf,u_f,dafdp,dufdp,PI],[PI_dot])
computeTheta = ca.Function('computeTheta',[x,u,p,x_r,k,xdotf,u_f,dafdp,dufdp,PI],[Theta])

# ComputeTubes
del_x = ca.SX.sym('del_x',nx,1)
del_u = ca.SX.sym('del_u',nu,1)
del_p = ca.SX.sym('del_p',npar,1)
W_par = ca.diag(del_p**2)

for i in range(nx):
    del_val = 0
    for j in range(npar):
       pert = ca.if_else(PI[i,j] >= 0,del_p[j],-del_p[j])
       del_val += PI[i,j]*pert
    del_x[i] = del_val

for i in range(nu):
    del_val = 0
    for j in range(npar):
       pert = ca.if_else(Theta[i,j] >= 0,del_p[j],-del_p[j])
       del_val += Theta[i,j]*pert
    del_u[i] = del_val


computeStateTube = ca.Function('computeStateTube',[PI,del_p],[del_x])
computeInputTube = ca.Function('computeInputTube',[x,u,p,x_r,k,xdotf,u_f,dafdp,dufdp,PI,del_p],[del_u])

"""
for i in range(nx):
    J = PI[i,:]@W_par@PI[i,:].T
    del_x[i] = ca.sqrt(J)

for i in range(nu):
    J = Theta[i,:]@W_par@Theta[i,:].T
    del_u[i] = ca.sqrt(J)
"""
