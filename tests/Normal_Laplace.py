from RobustGibbsObject import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import *
from tqdm import tqdm
import multiprocessing as mp


def display(dico,burnin=0,true_par=[]):
    par_names = list(dico["chains"].keys())
    
    f,ax= plt.subplots(2,len(par_names),figsize=(20,10))
    for i,par_name in enumerate(par_names):
        ax[0,i].plot(dico["chains"][par_name][burnin:])
        ax[0,i].set_title(par_name)
        sns.kdeplot(dico["chains"][par_name][burnin:],ax=ax[1,i])
        if true_par!=[]:
            ax[0,i].axhline(true_par[i],color="red")
            ax[1,i].axvline(true_par[i],color="red")
    plt.show()
from RobustGibbsObject.Distribution import Normal,Laplace,Gamma,InverseGamma
from RobustGibbsObject.Model import NormalModel,LaplaceModel, ReparametrizedLaplaceModel

# Bridge


from scipy.optimize import fixed_point
from scipy.special import logsumexp
def func_logBF_logsumexp(logBF,lratio_num,lratio_den):
    num_j = np.array([logsumexp([0,logBF+lratio_num[j]]) for j in range(len(lratio_num))])
    den_i = np.array([logsumexp([lratio_den[i],logBF]) for i in range(len(lratio_den))])
    num = logsumexp(-num_j)
    den = logsumexp(-den_i)
    return num-den

def logBF_logsumexp_fixedpoint(lratio_num,lratio_den):
    return fixed_point(func_logBF_logsumexp,0,args=(lratio_num,lratio_den))
from scipy.special import logsumexp
def logBF_Bridge_ratio(lratio_num, lratio_den,BF_init = 0, epsilon=0, a=0, n_iter = 10000, verbose = True):
    logBF = [np.inf,BF_init]
    n = min(len(lratio_num),len(lratio_den))
    k=0
    while np.abs(logBF[-1]-logBF[-2])>epsilon and k<n_iter:

        num_j = np.array([logsumexp([0,logBF[-1]+lratio_num[j]]) for j in range(n)])
        den_i = np.array([logsumexp([lratio_den[i],logBF[-1]]) for i in range(n)])
        
        num = logsumexp(-num_j)
        den = logsumexp(-den_i)

        new_logBF = num-den
        logBF.append(new_logBF)
        k+=1
        
        if verbose:print("Iteration {}: log(numerator) = {}, log(denominator) = {}, logBF = {}\n\n".format(k,num,den,new_logBF))
        
    return logBF[1:]
def func_r_paper(r,l1,l2,lstar):
    lstar = np.median(l1)
    return np.sum(np.exp(l2-lstar)/(np.exp(l2-lstar)+r))/np.sum(1/(np.exp(l1-lstar)+r))


def BF_paper(l1,l2):
    lstar = np.median(l1)
    r = fixed_point(func_r_paper,np.exp(-lstar),args=(l1,l2,lstar))
    BF = r*np.exp(lstar)
    return BF

def func_r_paper2(logBF,l1,l2):
    num_j = np.array([logsumexp([0,logBF-l2[j]]) for j in range(len(l2))])
    den_i = np.array([logsumexp([l1[i],logBF]) for i in range(len(l1))])
    num = logsumexp(-num_j)
    den = logsumexp(-den_i)
    return num-den


def logBF_paper(l1,l2):
    logBF = fixed_point(func_r_paper2,0,args=(l1,l2),maxiter=1000)
    return logBF
def log_ratio_normal_laplace(X,mu):
    return -(N/2)*np.log(np.pi)+np.sqrt(2)*np.sum(np.abs(X-mu))-np.sum((X-mu)**2)/2
# ABC Model Choice
def ABC_Model(T,N,med,MAD,epsilon):
    M = []
    Eta = []
    Theta = []
    Y = []
    eta_obs = np.array([med,MAD])
    for t in tqdm(range(T)):
        dist = epsilon+1
        while dist>epsilon:
            mean = np.random.normal(0,5,1)
            std = invgamma.rvs(1,1,1)
            if np.random.uniform()<0.5:
                y = np.random.normal(mean,std,N)
                m = 1
            else: 
                y = np.random.laplace(mean,std/np.sqrt(2),N)
                m = 2
            eta_sim = np.array([np.median(y),np.median(np.abs(y-np.median(y)))])
            var_sim = np.var(eta_sim,axis=0)
            dist = np.sum(((eta_obs-eta_sim)/varsim)**2)

        M.append(m)
        Theta.append([mean,std])
        Eta.append(eta_sim)
        Y.append(y)
    return {"M":np.array(M),"Theta":np.array(Theta),"Y":np.array(Y), "Eta":np.array(Eta)}
        
        
def ABC_Model_Paper(T,N,med,MAD,eps=.01):
    eta_obs = np.array([med,MAD])
    mean = np.random.normal(0,2,T)
    std = 1
    y_norm = norm(loc = mean[:T//2],scale = std).rvs([N,T//2]).T
    y_laplace =laplace(loc = mean[T//2:], scale = std/np.sqrt(2)).rvs([N,T//2]).T
    y = np.concatenate([y_norm,y_laplace],axis=0)
    eta_sim = np.array([np.median(y,axis=1),median_abs_deviation(y,axis=1)]).T
    dist = np.sum(((eta_obs-eta_sim)/np.std(eta_sim,axis=0))**2,axis=1)
    
    epsilon = np.quantile(dist,eps)
    idx = np.where(dist<epsilon)[0]
    model1 = len(idx[idx<T//2])
    model2 = len(idx[idx>=T//2])
    if model2 == 0:
        BF = np.inf
    else:
        BF = model1/model2    
    return {"Y":y,"eta_sim":eta_sim,"dist":dist,"BF":BF,"Proba1": model1/len(idx)}
        
        
# Comparison
from RobustGibbsObject.Model import NormalKnownScaleModel, LaplaceKnownScaleModel
from RobustGibbsObject.Distribution import Normal, Laplace
import pandas as pd

def Comparison(args):
    i,N,T = args
    X_norm = np.random.normal(0,1,N)
    X_laplace = np.random.laplace(0,1/np.sqrt(2),N)
    X_cauchy = np.random.standard_cauchy(N)
    L_Bridge = []
    L_logBridge = []
    L_ABC = []
    eps =.01
    for X in [X_norm,X_laplace,X_cauchy]:
        med, MAD = np.median(X),median_abs_deviation(X)

        ABC_BF = ABC_Model_Paper(T,N,med,MAD,eps)["BF"]
        L_ABC.append(ABC_BF)
        MCMC_norm = NormalKnownScaleModel(Normal(0,2)).Gibbs_med_MAD(T//2,N,med,MAD,List_X = True,verbose=False)
        MCMC_laplace = LaplaceKnownScaleModel(Normal(0,2)).Gibbs_med_MAD(T//2,N,med,MAD,List_X = True,verbose=False)
        X_Normal,X_Laplace = np.array(MCMC_norm["X"][1:]),np.array(MCMC_laplace["X"][1:])
        mu_Normal, mu_Laplace= np.array(MCMC_norm["chains"]["loc"]),np.array(MCMC_laplace["chains"]["loc"])
        
        l1 = [log_ratio_normal_laplace(X,mu) for X,mu in zip(X_Normal,mu_Normal)]
        l2 = [log_ratio_normal_laplace(X,mu) for X,mu in zip(X_Laplace,mu_Laplace)]
        
        BF_bridge = BF_paper(l1,l2)
        logBF_bridge = logBF_paper(l1,l2)
        
        L_Bridge.append(BF_bridge)
        L_logBridge.append(logBF_bridge)
    return [L_Bridge,L_logBridge,L_ABC]
    
        
        
        
res.shape
N_list = [11,101,1001]
T = 100000

n_iter = 100
Normal_Bridge_N, Normal_logBridge_N, Normal_ABC_N = [],[],[]
Laplace_Bridge_N, Laplace_logBridge_N, Laplace_ABC_N = [],[],[]
Cauchy_Bridge_N, Cauchy_logBridge_N, Cauchy_ABC_N = [],[],[]
for N in N_list:
    print("N = {}".format(N))
    pool = mp.Pool(mp.cpu_count())
    res = list(tqdm(pool.imap(Comparison, [(i,S,T) for i in range(n_iter)]),total = n_iter))
    pool.close()
    pool.join()

    #res = [Comparison(N,T) for _ in tqdm(range(n_iter))]
    res = np.array(res)
    Normal_Bridge_N.append(res[:,0,0])
    Normal_logBridge_N.append(res[:,1,0])
    Normal_ABC_N.append(res[:,2,0])
    Laplace_Bridge_N.append(res[:,0,1])
    Laplace_logBridge_N.append(res[:,1,1])
    Laplace_ABC_N.append(res[:,2,1])
    Cauchy_Bridge_N.append(res[:,0,2])
    Cauchy_logBridge_N.append(res[:,1,2])
    Cauchy_ABC_N.append(res[:,2,2])
    
# Create dataframes for each value of N
for i,N in enumerate(N_list):
    df_bridge_N = pd.DataFrame({'Normal': Normal_Bridge_N[i], 'Laplace': Laplace_Bridge_N[i], 'Cauchy': Cauchy_Bridge_N[i]})
    df_ABC_N = pd.DataFrame({'Normal': Normal_ABC_N[i], 'Laplace': Laplace_ABC_N[i], 'Cauchy': Cauchy_ABC_N[i]})
    df_logBridge_N = pd.DataFrame({'Normal': Normal_logBridge_N[i], 'Laplace': Laplace_logBridge_N[i], 'Cauchy': Cauchy_logBridge_N[i]})

    # Save dataframes to CSV files

    df_bridge_N.to_csv('Bridge_{}.csv'.format(N), index=False)
    df_ABC_N.to_csv('ABC_{}.csv'.format(N), index=False)
    df_logBridge_N.to_csv('logBridge_{}.csv'.format(N), index=False)
    