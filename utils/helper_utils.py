import numpy as np
import casadi as ca
import math 

def cross_prod(a, b):
    result = ca.vertcat(a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

    return result

def Rx(theta):
    R = ca.vertcat(ca.horzcat(1.0, 0.0, 0.0),
        ca.horzcat(0.0, ca.cos(theta), -ca.sin(theta)),
        ca.horzcat(0.0, ca.sin(theta), ca.cos(theta)))
    return R

def Ry(theta):
    R = ca.vertcat(ca.horzcat(ca.cos(theta), 0.0, ca.sin(theta)),
        ca.horzcat(0.0, 1.0, 0.0),
        ca.horzcat(-ca.sin(theta), 0.0, ca.cos(theta)))
    return R

def Rz(theta):
    R = ca.vertcat(ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
        ca.horzcat(ca.sin(theta),  ca.cos(theta), 0),
        ca.horzcat(0.0, 0.0, 1.0))
    return R


def skewMat(v):
    M = ca.vertcat(ca.horzcat(0.0, -v[2], v[1]),
                  ca.horzcat(v[2], 0.0, v[0]),
                  ca.horzcat(-v[1], v[0], 0.0))
    return M

def rpy2Rot(phi):
    R = Rz(phi[2]) @ Ry(phi[1]) @ Rx(phi[0])
    return R

def rpyDerivative(omega,rpy):
    theta_x = rpy[0]
    theta_y = rpy[1]
    theta_z = rpy[2]
    
    T = ca.vertcat(ca.horzcat(ca.cos(theta_y)*ca.cos(theta_z), -ca.sin(theta_z), 0),
                   ca.horzcat(ca.cos(theta_y)*ca.sin(theta_z), ca.cos(theta_z), 0),
                   ca.horzcat(-ca.sin(theta_y), 0, 1)) 
     
    rpy_dt = ca.solve(T,omega)

    return rpy_dt

def eSO3(Rd,R):
    E = 0.5 @ ((R.T@Rd) - (Rd.T@R))  
    eR =ca.vertcat((E[2, 1] - E[1, 2])/2,
    (E[0, 2] - E[2, 0])/2,
    (E[1, 0] - E[0, 1])/2)
    return eR
