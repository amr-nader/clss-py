
def computeTrajectory(qinit,qfin,t,total_time):
    if t>=total_time:
        t = total_time
    tau = t/total_time
    q_out = qinit + (qfin - qinit)*((6*tau**5)-(15*tau**4)+(10*tau**3))
    qdt_out = ((qfin - qinit)/total_time)*((30*tau**4)-(60*tau**3)+(30*tau**2))
    qddt_out = ((qfin - qinit)/(total_time**2))*((120*tau**3)-(180*tau**2)+(60*tau))
    return q_out,qdt_out,qddt_out
