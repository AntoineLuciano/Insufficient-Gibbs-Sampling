from typing import Dict, Tuple

from tqdm import tqdm
import numpy as np
from scipy.stats import norm, cauchy, lognorm, weibull_min,gamma,genpareto,median_abs_deviation,laplace, invgamma, iqr
# from InsufficientGibbs.Distribution import *
from Distribution import *
import seaborn as sns

def medMAD(X): return (np.median(X), median_abs_deviation(X))  

def display_chains(dico,burnin=0,true_par=[]):
    par_names = list(dico["chains"].keys())
    
    f,ax= plt.subplots(2,len(par_names),figsize=(20,10))
    for i,par_name in enumerate(par_names):
        sns.kdeplot(dico["chains"][par_name][burnin:],ax=ax[0,i])
        ax[0,i].set_xlabel(par_name, fontsize= 15)
        ax[0,i].set_ylabel("KDE density", fontsize= 15)
        ax[1,i].plot(dico["chains"][par_name][burnin:])
        ax[1,i].set_xlabel("Iterations", fontsize= 15)
        ax[1,i].set_ylabel(par_name, fontsize= 15)
        
        if true_par!=[]:
            ax[1,i].axhline(true_par[i],color="red")
            ax[0,i].axvline(true_par[i],color="red")
    plt.show()
    
class Model:
    
    """
    Base class for models

    Parameters
    __________
    self.parameters_dict : Dict
        Keys are internal names for the priors of the model
        e.g. 'loc', 'scale' for a Gaussian model. Values are the user 
        defined Distributions. Note key must not be identical to value.name
    """
    
    def __init__(self,parameters_dict: Dict[str, Distribution]) -> None:
        
        self.parameters_dict = self._check_dict(parameters_dict)
        self.par_X = []    
        self.distrib_name="" 
        self.par_names = list(self.parameters_dict.keys())
        self.init_method = "naive"

        
    

    def _check_dict(self, parameters_dict:Dict[str, Distribution]) -> Dict[str, Distribution]:
        # for key, value in self.parameters_dict.items():
            # if not isinstance(value, Distribution):
            #     raise ValueError(f'Input parameter "{key}" of "{self.__class__.__name__}" needs to be a Distribution (see InsufficientGibbs.distributions), but is of type {type(value)}.')
        return parameters_dict

    def _check_domain(self, X) -> None:
        minn, maxx = self.domain()
        f = lambda x: not(minn < x < maxx)
        if len(list(filter(f, X))) > 0:
            raise ValueError(f'some elements of X are not in the domain of the model, which is ({minn}, {maxx}).')

    def domain(self) -> None:
        """
        Should be overridden by all subclasses
        """
        raise NotImplementedError
    
    ### ------------------- POSTERIOR SAMPLING ------------------- ###
    def posterior(self,X, std_prop):
        """Function to sample from the posterior of parameters theta given data X."""
        
        def posterior_NIG(X,mu_0,nu,alpha,beta):
            N = len(X)
            M = (nu*mu_0+np.sum(X))/(nu+N)
            C = nu+N
            A = alpha+N/2
            B = beta + (np.sum((X-np.mean(X))**2)+N*nu/(nu+N)*(mu_0-np.mean(X))**2)/2
            
            sigma2 = invgamma(a = A, scale = B).rvs(1)[0]
            mu = norm(loc = M, scale = np.sqrt(sigma2/C)).rvs(1)[0]
            
            return {self.loc.name: mu, self.scale.name: np.sqrt(sigma2)}
        
        
        current_theta = self.parameters_value
        if self.distrib_name == 'normal': 
            if self.parameters_dict[self.loc.name].distrib_name =="normal" and self.parameters_dict[self.scale.name].distrib_name == "inverse_gamma":
                mu_0, sigma_0 = self.parameters_dict[self.loc.name].loc.value, self.parameters_dict[self.scale.name].scale.value
                alpha,beta = self.parameters_dict[self.scale.name].shape.value,self.parameters_dict[self.scale.name].scale.value
                current_theta = posterior_NIG(X,mu_0,1/sigma_0,alpha,beta)
                self._distribution= self.type_distribution(theta=list(current_theta.values()))
                return current_theta

        for param_name, param in self.parameters_dict.items():
            current_value = self.parameters_value[param_name]
            proposed_value = np.random.normal(current_value, std_prop[param_name])
            
            if not param._check_domain([proposed_value]):
                print("CONTINUE {}",proposed_value)
                continue

            current_llikelihood = self.type_distribution(theta=list(current_theta.values())).llikelihood(X)
            proposed_theta = current_theta.copy()
            proposed_theta[param_name]=proposed_value
            proposed_llikelihood = self.type_distribution(theta=list(proposed_theta.values())).llikelihood(X)
            
            current_lprior = param._distribution.logpdf(current_value)
            proposed_lprior = param._distribution.logpdf(proposed_value)
            
            ratio = np.exp(proposed_llikelihood - current_llikelihood + proposed_lprior - current_lprior)

            if np.random.uniform(0, 1) < ratio:
                current_theta[param_name]=proposed_value

        self._distribution= self.type_distribution(theta=list(current_theta.values()))
        return current_theta

    ### ------------------- QUANTILES ------------------- ###
    

    def log_order_stats_density(self, X, I):
        f,F= self._distribution.pdf,self._distribution.cdf
        
        return np.sum([np.log(F([X[i+1]]) - F(X[i]))*(I[i+1]-I[i]-1) for i in range(len(X)-1)]) + np.sum([np.log(f(X[i])) for i in range(1,len(X)-1)])

    
    
    def Init_X_Quantile(self, q_j, P, N, epsilon = .001):
        
        H_j = np.round(np.array(P) * (N - 1) + 1,8)
        I_j = np.floor(H_j)
        G_j = np.round(H_j - I_j, 8)

        I_j = [[I_j[i]] if G_j[i] ==0  else [I_j[i],I_j[i]+1] for i in range(len(I_j))]

        G = []
        Q = []
        I = []

        for i, I_j_i in enumerate(I_j):
            merged_I = False
            for j, I_j in enumerate(I):
                if any(x in I_j_i for x in I_j):
                    I[j] = sorted(list(set(I_j + I_j_i)))
                    G[j].append(G_j[i])
                    Q[j].append(q_j[i])
                    merged_I = True
                    break

            if not merged_I:
                I.append(I_j_i)
                G.append([G_j[i]])
                Q.append([q_j[i]])
        
        Q_tot = []
        Q_sim = []
        I_sim = []
        for k in range(len(I)):
            if G[k][0] == 0:
                Q_tot.append([Q[k][0]])
            else:
                Q_tot_k = [Q[k][0]-epsilon]
                Q_sim.append(Q[k][0]-epsilon)
                I_sim.append(I[k][0])
                for i in range(len(G[k])):
                    Q_new = (Q[k][i]-Q_tot_k[-1]*(1-G[k][i]))/(G[k][i])
                    if Q_tot_k[-1] > Q[k][i] or Q_new < Q_tot_k[-1] :
                        raise ValueError("Invalid quantile initialization")
                    Q_tot_k.append((Q[k][i]-Q_tot_k[-1]*(1-G[k][i]))/(G[k][i]))
                Q_tot.append(Q_tot_k)
        I_order = np.hstack(I)
        Trunc_eff = I_order[1:]-I_order[:-1]-1
        Trunc_eff = [I_order[0]-1]+list(Trunc_eff)+[N-I_order[-1]]
        X_order = np.hstack(Q_tot)
        Trunc_inter = [-np.inf]+list(X_order)+[np.inf]
        a,b =np.repeat(Trunc_inter[:-1],Trunc_eff), np.repeat(Trunc_inter[1:],Trunc_eff)

        X_trunc = self._distribution.truncated(
                    a=a,
                    b=b,
                    size=len(a),   
                )

        X_0 = np.round(np.append(X_trunc, X_order).reshape(-1), 8)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
    def Quantile_Init(self,  Q, P, N, theta_0 = {}, epsilon=0.001):
        if theta_0.keys()!= self.par_names:
            theta_0 = self.Init_theta_Quantile(Q, P, N, theta_0)
        self.parameters_value = theta_0
        self._distribution= self.type_distribution(theta=list(self.parameters_value.values()))
        return self.Init_X_Quantile(Q, P, N, epsilon=epsilon)

    
    def OrderStats_Quantile_MH(self, q_j, Q_tot, I, G, N, std_prop):
        f,Q = self._distribution.pdf,self._distribution.ppf
        I_sim = []
        Q_tot_star = []
        I = [[0]]+list(I)+[[N+1]]
        Q_sim=[]
        Q_tot = [[-np.inf]]+(Q_tot)+[[np.inf]]

        for j in range(1,len(Q_tot)-1):
            if len(Q_tot[j])==1:
                Q_tot_star.append(Q_tot[j])
            else:
                Q_tot_star_j = []
                I_sim = I[j][0]
                p = I_sim / (N + 1)
                Var_K = p * (1 - p) / ((N + 2) * f(Q(p)) ** 2)
                Std_Kernel = std_prop * np.sqrt(Var_K) / (1 - G[j-1][0])
                Q_tot_star_j = [np.random.normal(Q_tot[j][0], Std_Kernel)]
                if Q_tot_star_j[0] > q_j[j-1][0]: 
                    Q_sim.append(Q_tot_star_j[0])
                    continue
                
                for i,g in enumerate(G[j-1]):
                    Q_tot_star_j.append((q_j[j-1][i]-(1-g)*Q_tot_star_j[-1])/g)
        
                if np.min(Q_tot_star_j)< Q_tot[j-1][-1] or np.max(Q_tot_star_j)>Q_tot[j+1][0] or len(Q_tot[j])==1 or np.any(Q_tot_star_j[:-1] > Q_tot_star_j[1:]):
                    Q_sim.append(Q_tot[j][0])
                    continue
                X_current = [Q_tot[j-1][-1]]+Q_tot[j]+[Q_tot[j+1][0]]
                X_candidate = [Q_tot[j-1][-1]]+Q_tot_star_j+[Q_tot[j+1][0]]

                I_j = [I[j-1][-1]]+I[j]+[I[j+1][0]]

                log_density_current = self.log_order_stats_density(X_current, I_j) 
                log_density_candidate = self.log_order_stats_density(X_candidate, I_j)

                ratio = np.exp(log_density_candidate - log_density_current)

                if np.random.uniform(0, 1) < ratio:
                    Q_tot[j] = Q_tot_star_j
                Q_sim.append(Q_tot[j][0])
                
        return Q_tot[1:-1],Q_sim
    
    def Resample_X_Quantile(self, q_j, Q_tot, Q_sim, I, G, N, std_prop):
        if np.max([len(q_tot) for q_tot in Q_tot])>1:
            Q_tot,Q_sim = self.OrderStats_Quantile_MH(q_j, Q_tot, I, G,N,std_prop)
        I_order = np.hstack(I)
        Trunc_eff = I_order[1:]-I_order[:-1]-1
        Trunc_eff = [I_order[0]-1]+list(Trunc_eff)+[N-I_order[-1]]
        X_order = np.hstack(Q_tot)
        Trunc_inter = [-np.inf]+list(X_order)+[np.inf]
        a,b =np.repeat(Trunc_inter[:-1],Trunc_eff), np.repeat(Trunc_inter[1:],Trunc_eff)
        X_trunc = self._distribution.truncated(
                    a=a,
                    b=b,
                    size=len(a),   
                )

        X_0 = np.round(np.append(X_trunc, X_order).reshape(-1), 8)
        return X_0, Q_tot, Q_sim

    
    def Gibbs_Quantile(self,T: int,N: int,Q: list,P: list, theta_0 : dict = {}, std_prop_dict: dict = {}, std_prop_quantile: float =0.1, List_X: bool = False,  epsilon : float =0.001, verbose : bool =True):
                        
        """
        Gibbs sampler to sample from the posterior of model parameters given a sequence of quantiles.

        Args:
            T (int): Number of iterations.
            N (int): Size of the vector X.
            Q (list): Observed quantile value.
            P (float): Probability associated to the observed quantiles Q.
            theta_0 (dict): Initial value of the parameters. Default set to {}. 
            std_prop_dict (dict): Dictionary of standard deviations for the proposal distribution of each parameter. Default set to 0.1 for each parmeters. 
            std_prop_quantile (float): Standard deviation for the proposal distribution of the order statistics. Default set to 0.1.
            List_X (bool): If True, the latent vectors X are stored at each iteration and returned. Default set to False.
            epsilon (float): The distance between the initialized order statistics and the observed quantiles. Default set to 0.001.
            verbose (bool): If True, print the acceptance rate of the order statistics and the acceptance rate of the parameters. Default set to True.
            
            
        Returns:
            A dictionary containing:
                - X (list): List of latent vectors X at each iteration.
                - chains (dict): Dictionary of the chains of each parameter.
                - Q_sim (np.array): Simulated order statistics at each iteration.
                - Q_tot (np.array): All the considered order statistics at each iteration.
                 ... input parameters
        """
                 
                         
        if std_prop_dict == {}:
            std_prop_dict = {param_name: 0.1 for param_name in self.par_names}
        X, q, q_tot, q_sim, I, I_sim, G = self.Quantile_Init(Q, P, N, theta_0, epsilon=epsilon)
        
        theta_0 = self.parameters_value
        Chains = {par_name: [theta_0[par_name]] for par_name in self.par_names}
        X_list = [X]
        Q_Tot = [q_tot]
        Q_Sim = [q_sim]
        
        for i in tqdm(range(T), disable=not (verbose)):
            
            X,q_tot,q_sim = self.Resample_X_Quantile(q,q_tot,q_sim,I,G,N,std_prop_quantile)

            theta = self.posterior(X,std_prop_dict)

            for par_name in self.par_names:
                Chains[par_name].append(theta[par_name])

            Q_Tot.append(list(q_tot))
            Q_Sim.append(list(q_sim))
            if List_X:
                X_list.append(X)
        if not (List_X):
            X_list.append(X)
            
        if verbose:
            Q = np.array(Q_Sim).T
            for i in range(Q.shape[0]):
                q = Q[i]
                print(
                    "Acceptance rate of the order statistics ({}) = {:.2%}".format(
                        int(I_sim[i]), (len(np.unique(Q[i])) - 1) / Q.shape[1]
                    )
                )
            acceptation_rate = [
                (len(np.unique(Chains[par_name])) - 1) / T for par_name in self.par_names
            ]
            print("Acceptation rates MH :", end=" ")
            for i in range(len(self.par_names)):
                print("{} = {:.2%}".format(self.par_names[i], acceptation_rate[i]), end=" ")
            print()
            
        Q_Tot = np.array([np.hstack(q_tot).reshape(-1) for q_tot in Q_Tot])
        return {
            "X": X_list,
            "chains": Chains,
            "Q_sim": np.array(Q_Sim),
            "Q_tot": np.array(Q_Tot),
            "T": T,
            "N": N,
            "Q": Q,
            "P": P,
        }
        
        
    ### ------------------- MEDIAN AND MAD ------------------- ###
    
    
    
    def Init_X_med_MAD_naive(self, N, med, MAD):
        n = N // 2
        k = np.ceil(n / 2)

        if N % 2 == 0:
            return np.repeat(
                    [
                        med - 0.01 * MAD,
                        med + 0.01 * MAD,
                        med + MAD * 0.99,
                        med + MAD * 1.01,
                        med - 1.5 * MAD,
                        med - 0.5 * MAD,
                        med + 0.5 * MAD,
                        med + 1.5 * MAD,
                    ],
                    [1, 1, 1, 1, n - k, k - 1, n - k - 2, k - 1],
                )
        return np.repeat(
                [
                    med,
                    med + MAD,
                    med - 1.5 * MAD,
                    med - 0.5 * MAD,
                    med + 0.5 * MAD,
                    med + 1.5 * MAD,
                ],
                [1, 1, n - k + 1, k - 1, n - k, k - 1],
            )
        
    def Init_X_med_MAD_loc_scale_stable(self,N,med,MAD):
        Z = self._distribution.rvs(size=N)
        m_Z, s_Z = medMAD(Z)
        X_0 = np.round((Z - m_Z) / s_Z * MAD + med, 8)
        return X_0
    
    def Init_X_med_MAD(self, N, med, MAD, method):
        if method == "naive":
            return self.Init_X_med_MAD_naive(N, med, MAD)
        elif method == "stable":
            return self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        else:
            raise ValueError("Invalid initialization method: {} Please specify the initialization in the model instance!".format(method))
        
    
    def med_MAD_Init(self,N, med, MAD, theta_0):
        if theta_0.keys()!= self.par_names:
            theta_0 = self.Init_theta_med_MAD(med,MAD)
        self.parameters_value = theta_0
        self._distribution= self.type_distribution(theta=list(self.parameters_value.values()))
        return self.Init_X_med_MAD(N, med, MAD, self.init_method)
    
    def zone_even(self,xi, X, med=None, MAD=None):
        X = np.array(X)
        n = len(X) // 2

        if len(self.par_X) == 0:
            n = len(X) // 2
            X_s = np.sort(X)
            med1 = X_s[n - 1]
            med2 = X_s[n]
            S = np.abs(X - med)
            S_s = np.sort(S)
            MAD1, MAD2 = S_s[n - 1], S_s[n]

            [i_MAD1, i_MAD2] = np.argsort(S)[n - 1 : n + 1]
            Xmad1, Xmad2 = X[[i_MAD1, i_MAD2]]
            self.par_X= [MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2]
        self.par_X= np.round(self.par_X, 8)
        [MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2] = self.par_X
        

        xi = np.round(xi, 8)
        if xi == med1:
            return "med1"
        elif xi == med2:
            return "med2"
        elif xi == np.round(med + MAD1, 8):
            return "med-MAD1"
        elif xi == np.round(med - MAD1, 8):
            return "med-MAD1"
        elif xi == np.round(med + MAD2, 8):
            return "med+MAD2"
        elif xi == np.round(med - MAD2, 8):
            return "med-MAD2"
        elif xi < np.round(med - MAD2, 8):
            return 1
        elif np.round(med - MAD1, 8) < xi < med1:
            return 2
        elif med2 < xi < np.round(med + MAD1, 8):
            return 3
        elif xi > np.round(med + MAD2, 8):
            return 4
        else:
            print(
                "PAS DE ZONE pour {}, m-MAD2 = {}, m-MAD1 ={}, m1 = {}, m2 = {}, m+MAD1 = {}, m+MAD2 = {}".format(
                    xi, med - MAD2, med - MAD1, med1, med2, med + MAD1, med + MAD2
                )
            )
            
    def Resample_X_med_MAD_even(self,X,med,MAD,index=None):
        def sym(m, x):
            return 2 * m - x

        

        def zone_even_C_ab(xi, med1, med2, MAD1, MAD2):
            med = (med1 + med2) / 2
            if xi < med - MAD2:
                return med2, med + MAD1
            elif xi < med - MAD1:
                return med - MAD2, med - MAD1
            elif xi < med1:
                return med + MAD2, np.inf
            elif xi < med2:
                return med1, med2
            elif xi < med + MAD1:
                return -np.inf, med - MAD2
            elif xi < med + MAD2:
                return med + MAD1, med + MAD2
            else:
                return med - MAD1, med1

        def zone_even_S_ab(xi, med1, med2, MAD1, MAD2):
            med = (med1 + med2) / 2
            if xi < med - MAD2:
                return med + MAD2, np.inf
            elif xi < med - MAD1:
                return med + MAD1, med + MAD2
            elif xi < med1:
                return med2, med + MAD1
            elif xi < med2:
                return med1, med2
            elif xi < med + MAD1:
                return med - MAD1, med1
            elif xi < med + MAD2:
                return med - MAD2, med - MAD1
            else:
                return -np.inf, med - MAD2

        def zone_even_E_ab(xi, med1, med2, MAD1, MAD2):
            med, MAD = (med1 + med2) / 2, (MAD1 + MAD2) / 2
            if xi < med - MAD:
                return -np.inf, med - MAD
            elif xi < med1:
                return med - MAD, med1
            elif xi < med2:
                return med1, med2
            elif xi < med + MAD:
                return med2, med + MAD
            else:
                return med + MAD, np.inf

        def zone_even_ab(xi, med1, med2, MAD1, MAD2):
            med = (med1 + med2) / 2
            if xi < med - MAD2:
                return -np.inf, med - MAD2
            elif xi < med - MAD1:
                return med - MAD2, med - MAD1
            elif xi < med1:
                return med - MAD1, med1
            elif xi < med2:
                return med1, med2
            elif xi < med + MAD1:
                return med2, med + MAD1
            elif xi < med + MAD2:
                return med + MAD1, med + MAD2
            else:
                return med + MAD2, np.inf

        if index == None:
            index = np.random.choice(len(X), 2, replace=False)
        X = np.array(X)
        xij = np.round(X[index], 8)
        xi, xj = xij[0], xij[1]
        n = len(X) // 2
        if len(self.par_X) == 0:
            n = len(X) // 2
            X_s = np.sort(X)
            med1 = X_s[n - 1]
            med2 = X_s[n]
            S = np.abs(X - med)
            S_s = np.sort(S)
            MAD1, MAD2 = S_s[n - 1], S_s[n]
            [i_MAD1, i_MAD2] = np.argsort(S)[n - 1 : n + 1]
            Xmad1, Xmad2 = X[[i_MAD1, i_MAD2]]
            self.par_X= [MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2]

        self.par_X= np.round(self.par_X, 8)

        MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2 = np.round(self.par_X, 8)
        change_med = False
        change_MAD = False

        if sorted(xij) == [med1, med2]:
            case = "1"
            s3 = np.sort(np.abs(X - med))[2]

            a, b = med - s3, med + s3
            xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
            xnew2 = sym(med, xnew1)

            change_med = True
        elif sorted(xij) == sorted([Xmad1, Xmad2]):
            S = np.sort(np.abs(X - med))
            epsilon = np.minimum(MAD1 - S[n - 2], S[n + 1] - MAD2)
            if xi < med and xj < med:
                case = "2b"
                a, b = med - MAD2 - epsilon, med - MAD1 + epsilon
                if a >= b:
                    raise Exception("ERROR in med,MAD perturbation (case 2b) !")
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                xnew2 = sym(med - MAD, xnew1)

            elif xi > med and xj > med:
                case = "2a"
                a, b = med + MAD1 - epsilon, med + MAD2 + epsilon
                if a >= b:
                    raise Exception("ERROR in med,MAD perturbation (case 2a) !")

                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                xnew2 = sym(med + MAD, xnew1)
            else:
                case = "2c"
                a1, b1, a2, b2 = (
                    med - MAD2 - epsilon,
                    med - MAD1 + epsilon,
                    med + MAD1 - epsilon,
                    med + MAD2 + epsilon,
                )
                if a1 >= b1 or a2 >= b2:
                    raise Exception("ERROR in med,MAD perturbation (case 2c) !")

                xnew1 = self._distribution.truncated_2inter(a1=a1, a2=a2, b1=b1, b2=b2, size=1)[0]
                if xnew1 > med:
                    xnew2 = sym(med - MAD, sym(med, xnew1))
                else:
                    xnew2 = sym(med + MAD, sym(med, xnew1))
            change_MAD = True

        elif (med1 in xij or med2 in xij) and (Xmad1 in xij or Xmad2 in xij):
            case = "3"
            xnew1, xnew2 = xi, xj
        elif med1 in xij or med2 in xij:
            if xi in [med1, med2]:
                xm, xother = xi, xj
            elif xj in [med1, med2]:
                xm, xother = xj, xi
            else:
                raise Exception("ERROR in med,MAD perturbation (case 4) !")
            if xm == med1 and med + MAD1 > xother > med:
                case = "4a"
                a, b = med, med + MAD1
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                if xnew1 < med2:
                    xnew2 = sym(med, xnew1)
                    change_med = True
                else:
                    xnew2 = xm
            elif xm == med2 and med - MAD1 < xother < med:
                case = "4b"
                a, b = med - MAD1, med
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                if xnew1 > med1:
                    xnew2 = sym(med, xnew1)
                    change_med = True
                else:
                    xnew2 = xm
            else:
                case = "4c"
                a, b = zone_even_ab(xother, med1, med2, MAD1, MAD2)
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                xnew2 = xm
        elif Xmad1 in xij or Xmad2 in xij:
            if xi in [Xmad1, Xmad2]:
                xmad, xother = xi, xj
            elif xj in [Xmad1, Xmad2]:
                xmad, xother = xj, xi
            else:
                raise Exception("ERROR in med,MAD perturbation (case 5) !")
            if (xmad - med) * (xother - med) > 0 and (np.abs(xmad - med) - MAD) * (
                np.abs(xother - med) - MAD
            ) > 0:
                case = "5b "
                a, b = zone_even_ab(xother, med1, med2, MAD1, MAD2)
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                xnew2 = xmad
            elif (xmad - med) * (xother - med) > 0 and (np.abs(xmad - med) - MAD) * (
                np.abs(xother - med) - MAD
            ) < 0:
                case = "5a"
                a, b = zone_even_E_ab(xother, med1, med2, MAD1, MAD2)
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                if med - MAD2 <= xnew1 <= med - MAD1:
                    xnew2 = sym(med - MAD, xnew1)
                    change_MAD = True
                elif med + MAD1 <= xnew1 <= med + MAD2:
                    xnew2 = sym(med + MAD, xnew1)
                    change_MAD = True
                else:
                    xnew2 = xmad
            elif (xmad - med) * (xother - med) < 0 and (np.abs(xmad - med) - MAD) * (
                np.abs(xother - med) - MAD
            ) > 0:
                case = "5c "
                a1, b1 = zone_even_ab(xother, med1, med2, MAD1, MAD2)
                a2, b2 = zone_even_S_ab(xother, med1, med2, MAD1, MAD2)
                xnew1 = self._distribution.truncated_2inter(a1=a1, a2=a2, b1=b1, b2=b2, size=1)[0]
                if a2 <= xnew1 <= b2:
                    xnew2 = sym(med, xmad)
                    if xmad == Xmad1:
                        Xmad1 = sym(med, Xmad1)
                    elif xmad == Xmad2:
                        Xmad2 = sym(med, Xmad2)
                    else:
                        raise Exception("ERROR in med,MAD perturbation (case 5c) !")
                else:
                    xnew2 = xmad

            else:
                case = "5d "
                a, b = zone_even_E_ab(xother, med1, med2, MAD1, MAD2)
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                if med - MAD2 <= xnew1 <= med - MAD1:
                    xnew2 = sym(med + MAD, sym(med, xnew1))
                    change_MAD = True
                elif med + MAD1 <= xnew1 <= med + MAD2:
                    xnew2 = sym(med - MAD, sym(med, xnew1))
                    change_MAD = True
                else:
                    xnew2 = xmad

        else:
            l_zone = [
                self.zone_even(xi, X, med=med, MAD=MAD),
                self.zone_even(xj, X, med=med, MAD=MAD),
            ]
            sort_zone = sorted(l_zone)
            if sort_zone in [[1, 2], [3, 4]]:
                case = "6a "
                if xi < med:
                    a, b = -np.inf, med1
                else:
                    a, b = med2, np.inf
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                if med - MAD2 <= xnew1 <= med - MAD1:
                    xnew2 = sym(med - MAD, xnew1)
                    change_MAD = True
                elif med + MAD1 <= xnew1 <= med + MAD2:
                    xnew2 = sym(med + MAD, xnew1)
                    change_MAD = True
                elif xnew1 < med - MAD2:
                    a,b = med-MAD1, med1
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
                elif xnew1 > med + MAD2:
                    a,b = med2, med + MAD1
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
                elif med1 > xnew1 > med - MAD1:
                    a,b = -np.inf, med - MAD2
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
                else:
                    a,b = med + MAD2, np.inf
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
            elif sort_zone == [2, 3]:
                case = "6b "
                a, b = med - MAD1, med + MAD1
                xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                if med1 < xnew1 < med2:
                    xnew2 = sym(med, xnew1)
                    change_med = True
                elif xnew1 < med1:
                    a,b = med2, med + MAD1
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
                else:
                    a, b = med - MAD1, med1
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
            elif sort_zone in [[1, 3], [2, 4]]:
                case = "6c "
                a1, b1, a2, b2 = -np.inf, med1, med2, np.inf
                xnew1 = self._distribution.truncated_2inter(a1=a1, a2=a2, b1=b1, b2=b2, size=1)[0]
                if med - MAD2 <= xnew1 <= med - MAD1:
                    xnew2 = sym(med + MAD, sym(med, xnew1))
                    change_MAD = True
                elif med + MAD1 <= xnew1 <= med + MAD2:
                    xnew2 = sym(med - MAD, sym(med, xnew1))
                    change_MAD = True

                else:
                    a, b = zone_even_C_ab(xnew1, med1, med2, MAD1, MAD2)
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
            else:
                case = "6d"
                a1, b1 = zone_even_ab(xi, med1, med2, MAD1, MAD2)
                xnew1 = self._distribution.truncated(a=a1, b=b1,size=1)[0]
                a2, b2 = zone_even_ab(xj, med1, med2, MAD1, MAD2)
                xnew2 = self._distribution.truncated(a=a2, b=b2,size=1)[0]

        [xnew1, xnew2] = np.round([xnew1, xnew2], 8)
        X[index] = np.array([xnew1, xnew2]).reshape(-1)

        if change_med:
            [med1, med2] = sorted([xnew1, xnew2])

        if change_MAD:
            S_s = np.sort([np.abs(xnew1 - med), np.abs(xnew2 - med)])
            [MAD1, MAD2] = S_s.reshape(-1)

            [Xmad1, Xmad2] = np.array([xnew1, xnew2])[
                np.argsort([np.abs(xnew1 - med), np.abs(xnew2 - med)])
            ].reshape(-1)
            [i_MAD1, i_MAD2] = np.array(index)[
                np.argsort([np.abs(xnew1 - med), np.abs(xnew2 - med)])
            ]

        self.par_X= np.round(
            np.array([MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2]).squeeze(), 8
        )
        


        return X

    
    def Resample_X_med_MAD_odd(self,X,med,MAD,index=None):

        def zone_odd(xi, med, MAD):
            xi = np.round(xi, 8)

            if xi == med:
                return "med"
            elif xi == np.round(med + MAD, 8):
                return "med+MAD"
            elif xi == np.round(med - MAD, 8):
                return "med-MAD"
            elif xi < med - MAD:
                return 1
            elif xi < med:
                return 2
            elif xi < med + MAD:
                return 3
            else:
                return 4

        def zone_odd_C_ab(xi, med, MAD):
            xi = np.round(xi, 8)
            if xi == med:
                return med, med
            elif xi == np.round(med - MAD, 8):
                return np.round(med + MAD, 8), np.round(med + MAD, 8)
            elif xi == np.round(med + MAD, 8):
                return np.round(med - MAD, 8), np.round(med - MAD, 8)
            elif xi < med - MAD:
                return med, np.round(med + MAD, 8)
            elif xi < med:
                return np.round(med + MAD, 8), np.inf
            elif xi < med + MAD:
                return -np.inf, np.round(med - MAD, 8)
            else:
                return np.round(med - MAD, 8), med

        def zone_odd_ab(xi, med, MAD):
            xi = np.round(xi, 8)

            if xi == med:
                return med, med
            elif xi == np.round(med + MAD, 8):
                return np.round(med + MAD, 8), np.round(med + MAD, 8)
            elif xi == np.round(med - MAD, 8):
                return np.round(med - MAD, 8), np.round(med - MAD, 8)
            elif xi < med - MAD:
                return -np.inf, np.round(med - MAD, 8)
            elif xi < med:
                return np.round(med - MAD, 8), med
            elif xi < med + MAD:
                return med, np.round(med + MAD, 8)
            else:
                return np.round(med + MAD, 8), np.inf

        if index == None:
            index = np.random.choice(len(X), 2, replace=False)
        xij = np.round(X[index], 8)
        xij = X[index]
        xi, xj = xij[0], xij[1]
        a, b = 0, 0
        if len(self.par_X) == 0:
            if np.round(med + MAD, 8) in X:
                xmad = np.round(med + MAD, 8)
            elif np.round(med - MAD, 8) in X:
                xmad = np.round(med - MAD, 8)
            else:
                raise Exception("No MAD found!")
            i_MAD = np.where(X == xmad)[0][0]
            
            self.par_X= [i_MAD, float(xmad)]
        [i_MAD, xmad] = self.par_X
        med, xmad = np.round([med, xmad], 8)

        if med in xij and xmad in xij:
            case = "1"
            xnew1, xnew2 = xi, xj
        elif med in xij:
            case = "2"
            if xi == med:
                xother = xj
            elif xj == med:
                xother = xi
            else:
                print("Probleme m")
            a, b = zone_odd_ab(xother, med, MAD)
            xnew1, xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0], med
        elif xmad in xij:
            if xi == xmad:
                xother = xj
            elif xj == xmad:
                xother = xi
            else:
                print("Probleme xmad")
            if (xmad - med) * (xother - med) > 0:
                case = "3a"
                a, b = zone_odd_ab(xother, med, MAD)
                xnew1, xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0], xmad
            else:
                if np.abs(xother - med) > MAD:
                    a1, b1, a2, b2 = -np.inf,med - MAD, med + MAD, np.inf,
                    xnew1 = self._distribution.truncated_2inter(a1=a1, a2=a2, b1=b1, b2=b2, size=1)[0]
                    case = "3b"
                else:
                    case = "3c"
                    # case = "3c : {} devient {} dans R et {} devient {} dans [{},{}]".format(round(xi,3),round(xnew1,3),round(xj,3),round(xnew2,3),a,b)
                    a, b = med - MAD, med + MAD
                    xnew1 = self._distribution.truncated(a=a, b=b,size=1)[0]
                if xnew1 > med:
                    xmad = np.round(med - MAD, 8)
                else:
                    xmad = np.round(med + MAD, 8)
                xnew2 = xmad
                # print("xmad devient",xmad)
            i_MAD = index[1]
        else:
            if type(zone_odd(xi, med, MAD)) != str and type(zone_odd(xj, med, MAD)) != str:
                
                if sorted([zone_odd(xi, med, MAD), zone_odd(xj, med, MAD)]) in [
                    [1, 3],
                    [2, 4],
                ]:
                    case = "4a"
                    xnew1 = self._distribution.rvs(1)[0]
                    a, b = zone_odd_C_ab(xnew1, med, MAD)
                    xnew2 = self._distribution.truncated(a=a, b=b,size=1)[0]
                else:
                    case = "4b"
                    a1, b1 = zone_odd_ab(xi, med, MAD)
                    a2, b2 = zone_odd_ab(xj, med, MAD)
                    xnew1, xnew2 = self._distribution.truncated(a=a1, b=b1,size=1)[0], self._distribution.truncated(a=a2, b=b2,size=1)[0]
            else:
                print("zone_odd(xij)= ", zone_odd(xi, med, MAD), zone_odd(xj, med, MAD),self.par_X)
                raise Exception("ERROR in med,MAD perturbation !")
  
        # print(X[index],np.round(np.array([xnew1, xnew2]),8),case)
        X[index] = np.round(np.array([xnew1, xnew2]), 8).reshape(-1)
        self.par_X =[int(i_MAD), np.round(xmad,8)]
        return X
    
    def Resample_X_med_MAD(self, X,med,MAD):
        if len(X)%2==0:
            return self.Resample_X_med_MAD_even(X,med,MAD)
        return self.Resample_X_med_MAD_odd(X,med,MAD)
        
    
    def Gibbs_med_MAD(self,T: int,N: int,med: float,MAD: float, theta_0: dict = {}, std_prop_dict: dict = {}, List_X=False, verbose=True, True_X=[]):
        
        """
        Gibbs sampler to sample from the posterior of model parameters given the median and the MAD.

        Args:
            T (int): Number of iterations.
            N (int): Size of the vector X.
            med (float): Observed median.
            MAD (float): Observed MAD.
            theta_0 (dict): Dictionary of initial values for the parameters.
            std_prop_dict (dict): Dictionary of standard deviations for the proposal distribution of each parameter.
            List_X (bool): If True, the latent vectors X are stored at each iteration and returned.
            verbose (bool): If True, print the acceptance rate of the order statistics and the acceptance rate of the parameters.
            
        Returns:
        A dictionary containing:
                - X (list): List of latent vectors X at each iteration.
                - chains (dict): Dictionary of the chains of each parameter.
                 ... input parameters
        
        """
        
        if std_prop_dict == {}:
            std_prop_dict = {param_name: 0.1 for param_name in self.par_names}
        
        X = self.med_MAD_Init(N, med, MAD, theta_0)

        theta_0 = self.parameters_value
        Chains = {par_name: [theta_0[par_name]] for par_name in self.par_names}
        
        X_list = [X]
                
        for i in tqdm(range(T), disable=not (verbose)):
            if True_X==[]: X = self.Resample_X_med_MAD(X,med,MAD)
            else: X=True_X
            theta = self.posterior(X,std_prop_dict)
            for par_name in self.par_names:
                Chains[par_name].append(theta[par_name])
                
            if List_X: X_list.append(X)
        if not (List_X):
            X_list.append(X)

      
        if verbose :
            acceptation_rate = [
                (len(np.unique(Chains[par_name])) - 1) / T for par_name in self.par_names
            ]
            print("Acceptation rates MH :", end=" ")
            for i in range(len(self.par_names)):
                print("{} = {:.2%}".format(self.par_names[i], acceptation_rate[i]), end=" ")
            print()
        return {
            "X": X_list,
            "chains": Chains,
            "T": T,
            "N": N,
            "med": med,
            "MAD": MAD,
        
        }
    
    ### ------------------- MEDIAN AND IQR ------------------- ###

    ### Initialization
    
    def Init_X_med_IQR_naive(self, N, med, IQR, epsilon=0.001):
        n = N // 4
        q1 = med - IQR / 2
        q3 = med + IQR / 2

        if N % 4 == 1:
            X_0 = np.repeat([ q1, q3, med, med - 3 * IQR / 4, med - IQR / 4, med + IQR / 4, med + 3 * IQR / 4, ], [1, 1, 1, n, n - 1, n - 1, n], )          
            
        elif N % 4 == 3:
            g1, g3 = 1 / 2, 1 / 2
            q1a, q3a = q1 - epsilon * IQR, q3 - epsilon * IQR
            q3b = q3 + epsilon * IQR
            q1b = ((1 - g3) * q3a + g3 * q3b - (1 - g1) * q1a - IQR) / g1
            X_0 = np.repeat( [ q1a, q1b, q3a, q3b, med, med - 3 * IQR / 4, med - IQR / 4, med + IQR / 4, med + 3 * IQR / 4, ], [1, 1, 1, 1, 1, n, n - 1, n - 1, n], )
            
        elif N % 2 == 0:
            g1, g3 = 3 / 4, 1 / 4
            q1a, q3a = q1 - epsilon * IQR, q3 - epsilon * IQR
            q3b = q3 + epsilon * IQR
            q1b = ((1 - g3) * q3a + g3 * q3b - (1 - g1) * q1a - IQR) / g1
            med1, med2 = med - epsilon * IQR, med + epsilon * IQR
            X_0 = np.repeat( [ q1a, q1b, q3a, q3b, med1, med2, med - 3 * IQR / 4, med - IQR / 4, med + IQR / 4, med + 3 * IQR / 4, ], [1, 1, 1, 1, 1, 1, n - 1, n - 2, n - 2, n - 1])
        else:
            g1, g3 = 1 / 4, 3 / 4
            q1a, q3a = q1 - epsilon * IQR, q3 - epsilon * IQR
            q3b = q3 + epsilon * IQR
            q1b = ((1 - g3) * q3a + g3 * q3b - (1 - g1) * q1a - IQR) / g1
            med1, med2 = med - epsilon * IQR, med + epsilon * IQR
            X_0 = np.repeat( [ q1a, q1b, q3a, q3b, med1, med2, med - 3 * IQR / 4, med - IQR / 4, med + IQR / 4, med + 3 * IQR / 4, ], [1, 1, 1, 1, 1, 1, n, n - 2, n - 2, n], )
        return np.round(X_0, 8)
    
    def Init_X_med_IQR_loc_scale_stable(self,N,med,IQR):
        Z = self._distribution.rvs(size=N)
        m_Z, i_Z =  np.median(Z), iqr(Z)
        X_0 = np.round((Z - m_Z) / i_Z * IQR + med, 8)
        return X_0
    
    def Init_X_med_IQR(self,N,med,IQR,epsilon,method="naive"):
        if method=="naive":
            X_0 = self.Init_X_med_IQR_naive(N,med,IQR, epsilon)
        elif method=="stable":
            X_0 = self.Init_X_med_IQR_loc_scale_stable(N,med,IQR)
        else:
            raise Exception("Initialization method not recognized")
        print("N = {}".format(N))
        P = [0.25, 0.5, 0.75]
        H = np.array(P) * (N - 1) + 1
        I = np.floor(H).astype(int)
        G = np.round(H - I, 8)
        Q_tot = []
        I_order = []
        print("I = {}".format(I))
        X_0 = np.sort(X_0)
        for k in range(len(I)):
            if G[k] == 0:
                Q_tot.append(X_0[I[k] - 1])
                I_order.append(I[k])
            else:
                Q_tot.append(X_0[I[k] - 1])
                Q_tot.append(X_0[I[k]])
                I_order.append(I[k])
                I_order.append(I[k] + 1)
        if N % 4 == 1:
            Q_sim = [Q_tot[0]]
            I_sim = [I[0]]
        elif N % 4 == 3:
            Q_sim = [Q_tot[0], Q_tot[3], Q_tot[4]]
            I_sim = [I[0], I[2], I[2] + 1]
        else:
            Q_sim = [Q_tot[0], Q_tot[2], Q_tot[4], Q_tot[5]]
            I_sim = [I[0], I[1], I[2], I[2] + 1]

        I_order = np.array(I_order)
        # print("In Init_X_med_IQR, Q_sim = {} Q_tot = {} I_order = {} G = {} I_sim = {}".format(Q_sim, Q_tot, I_order, G, I_sim))
        return X_0, Q_sim, Q_tot, I_order, G, I_sim
    
    def med_IQR_Init(self, N, med, IQR, theta_0, epsilon):
        if theta_0.keys()!= self.par_names:
            theta_0 = self.Init_theta_med_IQR(med, IQR)
        self.parameters_value = theta_0
        self._distribution= self.type_distribution(theta=list(self.parameters_value.values()))
        return self.Init_X_med_IQR(N, med, IQR, epsilon, self.init_method)
    
        

    def OrderStats_med_IQR_MH(self, N, med, IQR, Q_sim, Q_tot, I_order, G, I_sim, std_prop):
        
        f,Q = self._distribution.pdf, self._distribution.ppf

        if N % 4 == 1:
            Norm = 1 / (1 - G[0])
        elif N % 4 == 3:
            Norm = np.array([1 / (1 - G[0]), 1 / G[2] / (1 - G[2]), 1 / (G[2])])
        else:
            Norm = np.array(
                [1 / (1 - G[0]), 1 / (1 - G[1]), 1 / G[2] / (1 - G[2]), 1 / (G[2])]
            )
        I_sim = np.array(I_sim)
        p = I_sim / (N + 1)
        Var_K = p * (1 - p) / ((N + 2) * f(Q(p)) ** 2)
        Std_Kernel = np.array(std_prop * np.sqrt(Var_K)) * Norm
        # print("Q_sim = {}, Std_Kernel = {}".format(Q_sim, Std_Kernel))    
        Q_sim_star_full = np.random.normal(Q_sim, Std_Kernel)

        for i in range(len(Q_sim)):
            Q_sim_star = Q_sim.copy()
            Q_sim_star[i] = Q_sim_star_full[i]
            if N % 4 == 1:
                Q_tot_star = [Q_sim_star[0], med, Q_sim_star[0] + IQR]
            elif N % 4 == 3:
                Q_tot_star = [ Q_sim_star[0], ( (1 - G[2]) * Q_sim_star[1] + G[2] * Q_sim_star[2] - (1 - G[0]) * Q_sim_star[0] - IQR ) / G[0], med, Q_sim_star[1], Q_sim_star[2], ]
            else:
                Q_tot_star = [ Q_sim_star[0], ( (1 - G[2]) * Q_sim_star[2] + G[2] * Q_sim_star[3] - (1 - G[0]) * Q_sim_star[0] - IQR ) / G[0], Q_sim_star[1], 2 * med - Q_sim_star[1], Q_sim_star[2], Q_sim_star[3], ]

            if (Q_tot_star == np.sort(Q_tot_star)).all():
                log_density_candidate = self.log_order_stats_density(Q_tot_star, I_order)
                log_density_current = self.log_order_stats_density(Q_tot, I_order)
                ratio = np.exp(log_density_candidate - log_density_current)
                if np.random.uniform(0, 1) < ratio:
                    Q_sim[i] = Q_sim_star_full[i]

            if N % 4 == 1:
                Q_tot = [Q_sim[0], med, Q_sim[0] + IQR]
            elif N % 4 == 3:
                Q_tot = [ Q_sim[0], ( (1 - G[-1]) * Q_sim[-2] + G[-1] * Q_sim[-1] - (1 - G[0]) * Q_sim[0] - IQR ) / G[0], med, Q_sim[-2], Q_sim[-1]]
            else:
                Q_tot = [ Q_sim[0], ( (1 - G[-1]) * Q_sim[-2] + G[-1] * Q_sim[-1] - (1 - G[0]) * Q_sim[0] - IQR ) / G[0], Q_sim[1], 2 * med - Q_sim[1], Q_sim[-2], Q_sim[-1] ]

        return Q_sim, Q_tot
    
    def Resample_X_med_IQR(self,N, med, IQR, Q_sim, Q_tot, I_order, G, I_sim, std_prop):
        # print("In Resample_X_med_IQR, Q_sim = {} Q_tot ={}".format(Q_sim ,Q_tot))
        Q_sim, Q_tot = self.OrderStats_med_IQR_MH(
            N, med, IQR, Q_sim, Q_tot, I_order, G, I_sim, std_prop
        )
        
        Trunc_eff = I_order[1:]-I_order[:-1]-1
        Trunc_eff = [I_order[0]-1]+list(Trunc_eff)+[N-I_order[-1]]
        X_order = np.array(Q_tot)
        Trunc_inter = [-np.inf]+list(X_order)+[np.inf]
        a,b =np.repeat(Trunc_inter[:-1],Trunc_eff), np.repeat(Trunc_inter[1:],Trunc_eff)
        X_trunc = self._distribution.truncated(
                    a=a,
                    b=b,
                    size=len(a),   
                )

        X_0 = np.round(np.append(X_trunc, X_order).reshape(-1), 8)
        return X_0, Q_sim, Q_tot

    def Gibbs_med_IQR(self,T: int, N: int, med: float, IQR: float, theta_0: dict = {}, std_prop_dict: dict = {}, std_prop_quantile=0.1,List_X=False, epsilon=0.001,verbose=True) -> dict:
        
        """Gibbs sampler for sampling from the posterior distribution of model parameters given the median and IQR of the data.

        Args:
            T (int): Number of iterations.
            N (int): Size of the vector X. 
            med (float): Observed median.
            IQR (float): Observed IQR (Interquartile Range).
            theta_0 (dict): Dictionary of initial values for the parameters.
            std_prop_dict (dict): Dictionary of standard deviations for the proposal distribution of each parameter.
            std_prop_quantile (float): Standard deviation for the proposal distribution of the order statistics.
            List_X (bool): If True, the latent vectors X are stored at each iteration and returned.
            epsilon (float): Small value to avoid numerical issues.
            verbose (bool): If True, print the acceptance rate of the order statistics and the acceptance rate of the parameters.
            

        Returns:
            A dictionary containing:
                chains (dict): The chains sampled from the parameters' posterior.
                X (list): List of latent vectors. 
                Q_sim (list): List of all simulated order statistics at each iteration. 
                Q_tot (list): List of all order statistics considered at each iteration.
                ... input parameters
            """ 
            
            
        if std_prop_dict == {}:
            std_prop_dict = {param_name: 0.1 for param_name in self.par_names}
        
        X, q_sim, q_tot, I_order, G, I_sim = self.med_IQR_Init(N, med, IQR, theta_0, epsilon=epsilon)
        
        theta_0 = self.parameters_value
        Chains = {par_name: [theta_0[par_name]] for par_name in self.par_names}
        
        X_list = [X]
        Q_Tot = [q_tot]
        Q_Sim = [q_sim]
        for i in tqdm(range(T), disable=not (verbose)):
            X,q_sim,q_tot = self.Resample_X_med_IQR(N, med, IQR, Q_Sim[-1], Q_Tot[-1], I_order, G, I_sim, std_prop_quantile)
            
            theta = self.posterior(X,std_prop_dict)

            for par_name in self.par_names:
                Chains[par_name].append(theta[par_name])

            Q_Tot.append(list(q_tot))
            Q_Sim.append(list(q_sim))
            if List_X:
                X_list.append(X)
        if not (List_X):
            X_list.append(X)
        if verbose:
            Q = np.array(Q_Sim).T
            for i in range(Q.shape[0]):
                q = Q[i]
                print(
                    "Acceptance rate of the order statistics ({}) = {:.2%}".format(
                        int(I_sim[i]), (len(np.unique(Q[i])) - 1) / Q.shape[1]
                    )
                )
        if verbose :
            acceptation_rate = [
                (len(np.unique(Chains[par_name])) - 1) / T for par_name in self.par_names
            ]
            print("Acceptation rates MH :", end=" ")
            for i in range(len(self.par_names)):
                print("{} = {:.2%}".format(self.par_names[i], acceptation_rate[i]), end=" ")
            print()
            
        Q_Tot = np.array([np.hstack(q_tot).reshape(-1) for q_tot in Q_Tot])
        return {
            "X": X_list,
            "chains": Chains,
            "Q_sim": np.array(Q_Sim),
            "Q_tot": np.array(Q_Tot),
            "T": T,
            "N": N,
            "med": med,
            "IQR": IQR,
        }