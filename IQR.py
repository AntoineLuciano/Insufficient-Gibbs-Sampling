import numpy as np
import scipy
from scipy.stats import norm,cauchy,weibull_min
from tqdm import tqdm

from Quantile import log_f_order_stats
from truncated import *

from normal_post import *
from cauchy_post import *
from weibull_post import *

def medIQR(X):
    return np.round([np.median(X),scipy.stats.iqr(X)],8)

# def MH_q1_med(q,med,iqr,N,mu,sigma,sigma_prop):
#     def f_q1_sachant_med(x,y,N,mu,sigma):
#         from scipy.stats import norm
#         f,F=norm(mu,sigma).pdf,norm(mu,sigma).cdf
#         if (x>=y):  return 0
#         i,j=N//4+1,N//2+1
#         return np.exp(np.log(f(x))+(j-i-1)*np.log(F(y)-F(x))+(i-1)*np.log(F(x)))
    
#     q_star=np.random.normal(q,sigma_prop)
#     if q_star>=med or q_star<med-iqr: return q
#     f_candidate,f_current=f_q1_sachant_med(q_star,med,N,mu,sigma),f_q1_sachant_med(q,med,N,mu,sigma)
#     ratio=f_candidate/f_current
#     if f_current==0: print(f_current)
#     if np.random.uniform()<=ratio  :
#         return q_star
#     return q

# def MH_q1_med2(q,med,iqr,N,loc,scale,sigma_prop,distribution,shape):
#     def f_q1_sachant_mi_MH(q,m,i,N,mean,std,shape):
#         if distribution=="normal":f,F=norm(loc=loc,scale=scale).pdf,norm(loc=loc,scale=scale).cdf
#         elif distribution=="cauchy":f,F=norm(loc=loc,scale=scale).pdf,norm(loc=loc,scale=scale).cdf
#         elif distribution[:7]=="weibull":f,F=weibull_min(shape,loc=loc,scale=scale).pdf,weibull_min(shape,loc=loc,scale=scale).cdf
#         k=N//4
#         #print("DANS F : q = {} med = {} iqr = {} N = {} mu = {} sigma = {}".format(q, med,iqr,N,mu,sigma))
#         return np.exp(np.log(f(q))+np.log(f(q+i))+(k-1)*(np.log(F(m)-F((q)))+np.log(F(q+i)-F(m)))+k*(np.log(F(q))+np.log(1-F(q+i))))
    
#     q_star=np.random.normal(q,sigma_prop)
#     if q_star>=med or q_star<=med-iqr: return q
#     #print("q = {} q*= {} med = {} iqr = {} N = {} mu = {} sigma = {}".format(q, q_star, med,iqr,N,mu,sigma))
#     f_candidate,f_current=f_q1_sachant_mi_MH(q_star,med,iqr,N,mu,sigma),f_q1_sachant_mi_MH(q,med,iqr,N,mu,sigma)
#     ratio=f_candidate/f_current
#     if f_current==0: print(f_current,q)
#     if np.random.uniform()<=ratio  :
#         return q_star
#     return q

# def X_m_iqr(X,m,iqr,mean,std,std_prop,distribution,shape=1):
#     N=len(X)
#     q1=MH_q1_med2(np.quantile(X,.25),m,iqr,N,mean,std,std_prop,distribution,shape)
#     q3=q1+iqr
    
#     k=N//4
#     a, b = np.repeat(
#         [-np.inf, q1, m, q3], [k, k - 1,  k-1, k ]
#     ), np.repeat([q1, m, q3, np.inf], [k, k - 1, k-1, k])
    
#     sample =truncated(
#         a=(a - mean) / std, b=(b - mean) / std, loc=np.repeat(mean,len(a)), scale=np.repeat(std,len(a)),size=len(a)
#     ,distribution=distribution)
#     return np.append(sample, [q1,q3,m]).reshape(-1),q1
    


def m_IQR_MH(m,IQR,Q_sim,Q_tot,N,loc,scale,K,G,I,distribution,std_prop,shape=1):          
    #print("Q_sim",Q_sim)
    log_density_current= log_f_order_stats(Q_tot,K,N,loc,scale,distribution)
    #print(Q_sim,Q_tot)
    if distribution=="normal":
        f,Q=norm(loc,scale).pdf,norm(loc,scale).ppf
        
    # elif distribution=="lognormal":
    #     f,Q=norm(loc,scale).pdf,norm(loc,scale).ppf
    #     Q_val,Q_sim,Q_tot=np.log(Q_val),np.log(Q_sim),np.log(Q_tot)
        # distribution="normal"
        
        # Q= lambda x: np.exp(norm(loc,scale).ppf(x))
        # f=lambda x: pdf_lognorm(x,loc,scale)
    elif distribution=="cauchy":
        f,Q=scipy.stats.cauchy(loc,scale).pdf,scipy.stats.cauchy(loc,scale).ppf
        
    elif distribution[:7]=="weibull":
        f,Q=weibull_min(shape,loc=loc,scale=scale).pdf,weibull_min(shape,loc=loc,scale=scale).ppf



    if N%4==1:  
        I_sim = [I[0]]
        Norm=1/(1-G[0])
        Tot_to_sim=[0]
    elif N%4==3: 
        I_sim = [I[0],I[2],I[2]+1]
        Norm=np.array([1/(1-G[0]),1/G[2]/(1-G[2]),1/(G[2])])
        Tot_to_sim=[0,2,3]
    else :  
        I_sim = [I[0],I[1],I[2],I[2]+1]
        Norm=np.array([1/(1-G[0]),1/(1-G[1]),  1/G[2]/(1-G[2]),1/(G[2])])
        Tot_to_sim=[0,1,2,3]
    I_sim=np.array(I_sim)
    p=I_sim/(N+1)
    Var_K=p*(1-p)/((N+2)*f(Q(p))**2)
    Std_Kernel=np.array(std_prop*np.sqrt(Var_K))*Norm
    #print(len(Q_sim),len(Std_Kernel))
    Q_sim_star_full= np.random.normal(Q_sim,Std_Kernel)
    #print(Q_sim_star)
    # if N%4==1: 
    #     Q_tot_star = [Q_sim_star[0],m,Q_sim_star[0]+IQR]
    # elif N%4==3:
    #     Q_tot_star = [Q_sim_star[0],((1-G[-1])*Q_sim_star[-2]+G[-1]*Q_sim_star[-1]-(1-G[0])*Q_sim_star[0]-IQR)/G[0],m,Q_sim_star[-2],Q_sim_star[-1]]
    # else : 
    #     Q_tot_star = [Q_sim_star[0],((1-G[-1])*Q_sim_star[-2]+G[-1]*Q_sim_star[-1]-(1-G[0])*Q_sim_star[0]-IQR)/G[0],Q_sim_star[1],2*m-Q_sim_star[1],Q_sim_star[-2],Q_sim_star[-1]]
    #print(Q_sim_star,Q_tot_star)

    for i in range(len(Q_sim)):
        Q_sim_star=Q_sim.copy()
        Q_sim_star[i]=Q_sim_star_full[i]
        if N%4==1: 
            Q_tot_star = [Q_sim_star[0],m,Q_sim_star[0]+IQR]
        elif N%4==3:
            Q_tot_star = [Q_sim_star[0],((1-G[2])*Q_sim_star[1]+G[2]*Q_sim_star[2]-(1-G[0])*Q_sim_star[0]-IQR)/G[0],m,Q_sim_star[1],Q_sim_star[2]]
        else : 
            Q_tot_star = [Q_sim_star[0],((1-G[2])*Q_sim_star[2]+G[2]*Q_sim_star[3]-(1-G[0])*Q_sim_star[0]-IQR)/G[0],Q_sim_star[1],2*m-Q_sim_star[1],Q_sim_star[2],Q_sim_star[3]]
        
        
        if (Q_tot_star==np.sort(Q_tot_star)).all():
            log_density_candidate= log_f_order_stats(Q_tot_star,K,N,loc,scale,distribution)
            log_density_current= log_f_order_stats(Q_tot,K,N,loc,scale,distribution)
            ratio=np.exp(log_density_candidate-log_density_current)
            if np.random.uniform(0,1)<ratio:
                Q_sim[i]=Q_sim_star_full[i]
                #print("Quantile simulé {} : Accepté {} Ratio {}".format(i,Q_sim_star[i],ratio))
               
            #else: print("Quantile simulé {} : Rejeté {} Ratio {}".format(i,Q_sim_star[i],ratio))
        #else: print("Quantile simulé {} : Rejeté Ordre {}".format(i,Q_tot_star))
        if N%4==1: 
            Q_tot = [Q_sim[0],m,Q_sim[0]+IQR]
        elif N%4==3:
            Q_tot = [Q_sim[0],((1-G[-1])*Q_sim[-2]+G[-1]*Q_sim[-1]-(1-G[0])*Q_sim[0]-IQR)/G[0],m,Q_sim[-2],Q_sim[-1]]
        else : 
            Q_tot = [Q_sim[0],((1-G[-1])*Q_sim[-2]+G[-1]*Q_sim[-1]-(1-G[0])*Q_sim[0]-IQR)/G[0],Q_sim[1],2*m-Q_sim[1],Q_sim[-2],Q_sim[-1]]
        
    return Q_sim,Q_tot
    #print(np.array(Q_tot_star),np.sort(Q_tot_star))
    

    # if (Q_tot_star!=np.sort(Q_tot_star)).any(): return Q_sim,Q_tot,log_density_current,"ordre rejet"
    # log_density_candidate= log_f_order_stats(Q_tot_star,K,N,loc,scale,distribution)
    # ratio=np.exp(log_density_candidate-log_density_current)
    # #print("ratio = {}".format(ratio))
    # if np.random.uniform(0,1)<ratio:
    #     #print("accept")
    #     return Q_sim_star,Q_tot_star,log_density_candidate,"accept"
    # #print("reject")
    # return Q_sim,Q_tot,log_density_current,"ratio rejet"


def X_m_IQR(m,IQR,Q_sim,Q_tot,N,loc,scale,K,G,I,distribution,std_prop,shape=1):
             
    Q_sim,Q_tot=m_IQR_MH(m,IQR,Q_sim,Q_tot,N,loc,scale,K,G,I,distribution,std_prop,shape=shape)

    K1=[K[0]-1]+list(K[1:]-K[:-1]-1)+[N-K[-1]]
    #print(K1)
    X1=np.insert(np.array(Q_tot).astype(float),0,-np.inf)

    X2=np.append(Q_tot,np.inf)
    
    a, b = np.repeat(
        X1, K1
    ), np.repeat(X2, K1)
    
    sample =truncated(
        a=(a - loc) / scale, b=(b - loc) / scale, loc=np.repeat(loc,len(a)), scale=np.repeat(scale,len(a)),size=len(a), distribution=distribution,shape=shape)
    return np.round(np.append(sample, Q_tot).reshape(-1),8),Q_sim,Q_tot


def IQR_init(N, med, IQR, distribution,init_X=[]):
    loc,scale,shape=0,1,1
    if init_X!=[]:
        X_0=np.sort(init_X)
    else:
        if distribution == "normal": Y = np.round(np.random.normal(loc, scale, N), 8)
        elif distribution == "cauchy": Y = np.round(scipy.stats.cauchy(loc=loc, scale=scale).rvs(N), 8)
        elif distribution[:7] == "weibull": Y= np.round(weibull_min(c=shape,loc=loc,scale=scale).rvs(N), 8)
        else : raise "UKNOWN DISTRIBUTION"
        m_Y, s_Y = np.median(Y), scipy.stats.iqr(Y)
        X_0= np.sort(np.round((Y - m_Y) / s_Y * IQR+ med, 8))
    if distribution == "normal":
        init_par = [med,IQR/(2*norm(loc=loc, scale=scale).ppf(0.75))]
    elif distribution == "cauchy":
        init_par=[med,IQR/2]
    elif distribution=="weibull":
        init_par=[(loc-m_Y)/ s_Y * IQR+med,scale / s_Y * IQR,shape]
    elif distribution=="weibull2":
        init_par=[0,scale / s_Y * IQR,shape]
    
    P=[0.25,.5,.75]
    H=np.array(P)*(N-1)+1
    I=np.floor(H).astype(int)
    G=np.round(H-I,8)
    Q_tot=[]
    K=[]

    for k in range(len(I)): 
        if G[k]==0: 
            Q_tot.append(X_0[I[k]-1])
            K.append(I[k])
        else: 
            Q_tot.append(X_0[I[k]-1])
            Q_tot.append(X_0[I[k]])
            K.append(I[k])
            K.append(I[k]+1)
    if N%4==1: Q_sim=[Q_tot[0]]
    elif N%4 == 3:  Q_sim=[Q_tot[0],Q_tot[3],Q_tot[4]]
    else: Q_sim=[Q_tot[0],Q_tot[2],Q_tot[4],Q_tot[5]]
    #print(Q_tot)
    
    K=np.array(K)
    return X_0,init_par,Q_sim,Q_tot,K,G,I

def Gibbs_med_IQR(T,N,med,IQR,n_chains,distribution,par_prior=[0,1,1,1,1,1],std_prop1=0.1,std_prop2=0.1,std_prop3=0.1,std_prop_quantile=.1,List_X=False,verbose=True,perturb=True,init_X=[]):
    
    X_0,init_theta,q_sim,q_tot,K,G,I= IQR_init(N,med,IQR,distribution,init_X=init_X)
    if distribution=="weibull2":
        init_theta[0]=0 
    Theta=[init_theta]
    X_list=[X_0]
    Mean=[np.mean(X_0)]
    Std=[np.std(X_0)]
    Q_Tot=[q_tot]
    Q_Sim=[q_sim]


    if distribution=="weibull" or distribution=="weibull2":
        loc,scale,shape=init_theta
    else:
        loc,scale=init_theta
        shape=1
    print(init_theta,medIQR(X_0),q_tot,q_sim,np.quantile(X_0,[0.25,0.5,0.75]))
    X=X_0.copy()
    for i in tqdm(range(T),disable=not(verbose)):
        
        if perturb:X,q_sim,q_tot=X_m_IQR(med,IQR,q_sim,q_tot,N,Theta[-1][0],Theta[-1][1],K,G,I,distribution,std_prop_quantile,shape=shape)
                    
        
        if distribution=="normal":
            mu,tau=post_NG(X,par_prior)
            theta=[mu,1/np.sqrt(tau)]
        elif distribution=="lognormal":
            mu,tau=post_NG(np.log(X),par_prior)
            #print(np.std(np.log(X)),np.sqrt(1/tau))
            theta=[mu,np.sqrt(1/tau)]
        elif distribution=="cauchy":
            loc=post_cauchy_theta(Theta[-1][0],Theta[-1][1],X,par_prior[:2],std_prop1)
            scale=post_cauchy_gamma(loc,Theta[-1][1],X,par_prior[2:],std_prop2)
            theta=[loc,scale]
        
        elif distribution[:7]=="weibull":
            #print(np.min(X),loc)
            if distribution=="weibull":loc=post_weibull_loc(Theta[-1][0],Theta[-1][1],Theta[-1][2],X,par_prior[:2],std_prop1)
            else:loc=0
            scale=post_weibull_scale(loc,Theta[-1][1],Theta[-1][2],X,par_prior[2:4],std_prop2)
            shape=post_weibull_k(loc,scale,Theta[-1][2],X,par_prior[4:],std_prop3)
            theta=[loc,scale,shape]
            
        Theta.append(theta)
        Mean.append(np.mean(X))
        Std.append(np.std(X))
        Q_Tot.append(list(q_tot))
        Q_Sim.append(list(q_sim))
        if List_X: X_list.append(X.copy())
        
    if not(List_X): X_list.append(X.copy())
    if verbose: 
        Q=np.array(Q_Sim).T
        print(Q.shape)
        for i in range(Q.shape[0]):
            q=Q[i]
            print("Acceptance rate of Q {} = {:.2%}".format(i,len(np.unique(q))/len(q)))
    if verbose:print("Acceptance rate of Q =  {:.2%}".format((len(np.unique(Q_Sim,axis=0))-1)/len(Q_Sim)))

        
    if verbose and distribution=="cauchy":
        print("Acceptation rate of loc = {:.2%} and of scale = {:.2%}".format(len(np.unique(np.array(Theta)[:,0],axis=0))/len(Theta),len(np.unique(np.array(Theta)[:,1],axis=0))/len(Theta)))
    if verbose and distribution=="weibull":
        print("Acceptation rate of loc = {:.2%}, of scale = {:.2%} and of shape = {:.2%}".format(len(np.unique(np.array(Theta)[:,0],axis=0))/len(Theta),len(np.unique(np.array(Theta)[:,1],axis=0))/len(Theta),len(np.unique(np.array(Theta)[:,2],axis=0))/len(Theta)))
    return {"X":X_list,"Mean":Mean,"Std":Std,"chains":np.array(Theta).T,"Q_sim":np.array(Q_Sim),"Q_tot":np.array(Q_Tot),"I":I,"K":K,"G":G,"par_prior":par_prior,"distribution":distribution,"T":T,"N":N,"med":med,"IQR":IQR}