from typing import Dict, Tuple

from tqdm import tqdm
import numpy as np
from scipy.stats import norm, cauchy, lognorm, weibull_min,gamma,genpareto,median_abs_deviation,laplace


from RobustGibbsObject.Distribution import Distribution, Normal, Normal_known_scale, Cauchy, Gamma, LogNormal, TranslatedLogNormal, Weibull, Laplace, Laplace_known_scale,TranslatedWeibull, ReparametrizedTranslatedLogNormal, GeneralizedPareto,ParetoType2, ReparametrizedGeneralizedPareto,ReparametrizedParetoType2,FixedTranslatedLogNormal,FixedGeneralizedPareto,FixedReparametrizedTranslatedLogNormal,FixedReparametrizedGeneralizedPareto,FixedReparametrizedParetoType2, ReparametrizedLaplace, ReparametrizedGamma, ReparametrizedTranslatedGamma, FixedReparametrizedTranslatedGamma,ReparametrizedLogNormal
def medMAD(X):
    return (np.median(X), median_abs_deviation(X))

class Model:
    
    """
    Base class for quantile matching (Model) models

    Parameters
    __________
    self.parameters_dict : Dict
        Keys are internal names for the priors of the model
        e.g. 'loc', 'scale' for a Gaussian modek. Values are the user 
        defined Distributions. Note key must not be identical to value.name.
    """
    
    def __init__(self,parameters_dict: Dict[str, Distribution]) -> None:
        self.parameters_dict = self._check_dict(self.parameters_dict)
        self.model = None
        self.par_X = []

    def _check_dict(self, parameters_dict:Dict[str, Distribution]) -> Dict[str, Distribution]:
        # print(self.parameters_dict)
        for key, value in self.parameters_dict.items():
            if not isinstance(value, Distribution):
                raise ValueError(f'Input parameter "{key}" of "{self.__class__.__name__}" needs to be a Distribution (see RobustGibbs.distributions), but is of type {type(value)}.')

    def _check_domain(self, X) -> None:
        minn, maxx = self.domain()
        #print(X)
        f = lambda x: not(minn < x < maxx)
        if len(list(filter(f, X))) > 0:
            #print(len(list(filter(f, X))))
            raise ValueError(f'some elements of X are not in the domain of the model, which is ({minn}, {maxx}).')

    def domain(self) -> None:
        """
        Should be overridden by all subclasses
        """
        raise NotImplementedError
    
    
    ### QUANTILE FUNCTIONS 
    

    def log_order_stats_density(self, X, I):
        f,F= self._distribution.pdf,self._distribution.cdf
        
        return np.sum([np.log(F([X[i+1]]) - F(X[i]))*(I[i+1]-I[i]-1) for i in range(len(X)-1)]) + np.sum([np.log(f(X[i])) for i in range(1,len(X)-1)])
        #print("In log_order_stats_density : X = {} I = {}".format(X,I))
        # print(X)
        # print(F([X[1]]) - F(X[0]),F([X[3]]) - F(X[2]),f(X[2]),f(X[1]))
        # print(np.log(F([X[1]]) - F(X[0])),np.log(F([X[3]]) - F(X[2])),np.log(f(X[2])),np.log(f(X[1])))
        return (
            np.log(F([X[1]]) - F(X[0])) * (I[1] - I[0] - 1)
            + np.log(F([X[3]]) - F(X[2])) * (I[3] - I[2] - 1)
            + np.log(f(X[2]))
            + np.log(f(X[1]))
        )

    
    
    
    def Init_X_Quantile(self, q_j, P, N, epsilon = .001):
        
        H_j = np.round(np.array(P) * (N - 1) + 1,8)
        I_j = np.floor(H_j)
        G_j = np.round(H_j - I_j, 8)
        I_diff = I_j[1:]-I_j[:-1]
        I_j = [[I_j[i]] if G_j[i] ==0  else [I_j[i],I_j[i]+1] for i in range(len(I_j))]
        # G = []
        # Q = []
        # I = []
        
        # j=0
        # while j<len(I_j):
        #     if G_j[j] == 0:
        #         G.append([G_j[j]])
        #         Q.append([q_j[j]])
        #         I.append([I_j[j]])
        #     else:
        #         g_j,q_jbis,i_j= [G_j[j]],[q_j[j]],[I_j[j]]
        #         while I_diff[j] == 1:
        #             g_j.append(G_j[j+1])
        #             q_jbis.append(q_j[j+1])
        #             i_j.append(I_j[j+1])
        #             j+=1
        #         G.append(g_j)
        #         Q.append(q_jbis)
        #         I.append(i_j)
        # print("I = {} G = {} Q = {}".format(I,G,Q))
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
        print("I = {} G = {} Q = {}".format(I,G,Q))
        
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
    
    def OrderStats_MH(self, q_j, Q_tot, I, G,N,std_prop):
        f,Q = self._distribution.pdf,self._distribution.ppf
        I_sim = []
        Q_tot_star = []
        I = [[0]]+list(I)+[[N+1]]
        Q_sim=[]
        Q_tot = [[-np.inf]]+(Q_tot)+[[np.inf]]
        #print("len(q_j) = {} len(Q_tot) = {} len(I) = {} len(G) = {}".format(len(q_j),len(Q_tot),len(I),len(G)))
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
                #print("I_sim ",I[j][0])
                I_j = [I[j-1][-1]]+I[j]+[I[j+1][0]]
                #print("I_j ",I_j)
                
                #print("X_current = {} X_candidate = {}".format(X_current,X_candidate))
                log_density_current = self.log_order_stats_density(X_current, I_j) 
                log_density_candidate = self.log_order_stats_density(X_candidate, I_j)
                #print("log_density_current = {} log_density_candidate = {}".format(log_density_current,log_density_candidate))
                ratio = np.exp(log_density_candidate - log_density_current)
                #print("ratio = ",ratio)
                if np.random.uniform(0, 1) < ratio:
                    Q_tot[j] = Q_tot_star_j
                Q_sim.append(Q_tot[j][0])
                
        return Q_tot[1:-1],Q_sim
    
    def Resample_X_Q(self, q_j, Q_tot, Q_sim,N, I, G, std_prop):
        if np.max([len(q_tot) for q_tot in Q_tot])>1:
            Q_tot,Q_sim = self.OrderStats_MH(q_j, Q_tot, I, G,N,std_prop)
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

    # def Init_X_Quantile2(self, Q, P, N, epsilon=0.001):
    #     H = np.array(P) * (N - 1) + 1
    #     I = np.floor(H)
    #     G = np.round(H - I, 8)
    #     Q_sim = []
    #     Q_tot = []
    #     K = []
    #     for k in range(len(G)):
    #         K.append(I[k])
    #         if G[k] == 0:
    #             Q_tot.append(Q[k])
    #         else:
    #             Q_sim.append(Q[k] - epsilon)
    #             Q_tot.append(Q[k] - epsilon)
    #             Q_tot.append((Q[k] - Q_tot[-1] * (1 - G[k])) / G[k])
    #             K.append(I[k] + 1)
    #             if k < len(G) - 1:
    #                 if Q_tot[-1] > Q[k + 1]:
    #                     raise Exception("Initialization problem !")
    #     K = np.array(K)
    #     K1 = [K[0] - 1] + list(K[1:] - K[:-1] - 1) + [N - K[-1]]
    #     X1 = np.insert(np.array(Q_tot).astype(float), 0, -np.inf)

    #     X2 = np.append(Q_tot, np.inf)
    #     a, b = np.repeat(X1, K1), np.repeat(X2, K1)

    #     sample = self._distribution.truncated(
    #         a=a,
    #         b=b,
    #         size=len(a),   
    #     )

    #     X_0 = np.round(np.append(sample, Q_tot).reshape(-1), 8)
    #     return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
    # def OrderStats_MH2(self, Q_obs, Q_sim, Q_tot, N, K, I, G, std_prop):
    #     #print("In OrderStats_MH : theta = {}".format(theta))
    #     #print("MH on order statistics...")
    #     f,Q = self._distribution.pdf,self._distribution.ppf

    #     I_sim = np.array(I[np.where(G > 0)])
    #     p = I_sim / (N + 1)
    #     Var_K = p * (1 - p) / ((N + 2) * f(Q(p)) ** 2)
    #     Std_Kernel = std_prop * np.sqrt(Var_K) / (1 - G[np.where(G > 0)])
    #     Q_sim_star = np.random.normal(Q_sim[: len(Std_Kernel)], Std_Kernel)
    #     #print("len(Q_sim) = {} len(Q_sim_star) = {}".format(len(Q_sim),len(Q_sim_star)))

    #     Q_tot_star = []

    #     j = 0
    #     for i in range(len(Q_obs)):
    #         if G[i] > 0:
    #             Q_tot_star.append(Q_sim_star[j])
    #             Q_tot_star.append((Q_obs[i] - Q_sim_star[j] * (1 - G[i])) / G[i])
    #             j += 1
    #         else:
    #             Q_tot_star.append(Q_obs[i])

    #     Q_tot_star2 = np.array(Q_tot_star)
    #     Q_tot_star2 = np.insert(Q_tot_star2, 0, -np.inf)
    #     Q_tot_star2 = np.append(Q_tot_star2, np.inf)
    #     Q_tot2 = np.array(Q_tot)
    #     Q_tot2 = np.insert(Q_tot2, 0, -np.inf)
    #     Q_tot2 = np.append(Q_tot2, np.inf)
    #     K1 = np.array(K)
    #     K1 = np.insert(K1, 0, 0)
    #     K1 = np.append(K1, N + 1)
    #     i = 0
    #     j = 1
    #     k = 0

    #     while j < len(Q_tot2) - 1:
    #         if k >= len(G):
    #             print(
    #                 "ERREUR : k = ",
    #                 k,
    #                 " len(G) = ",
    #                 len(G),
    #                 "len(Q_sim*)=",
    #                 len(Q_sim_star),
    #             )
    #         if G[k] > 0:
    #             if Q_sim_star[i] < Q_tot2[j - 1] or Q_sim_star[i] > Q_obs[k]:
    #                 j += 2
    #                 i += 1
    #                 k += 1
    #                 #print("Q{} order problem!".format(k))
    #                 continue

    #             X_current = Q_tot2[j - 1 : j + 3]
    #             X_candidate = [
    #                 Q_tot2[j - 1],
    #                 Q_tot_star2[j],
    #                 Q_tot_star2[j + 1],
    #                 Q_tot2[j + 2],
    #             ]
    #             I_i = K1[j - 1 : j + 3]
    #             #print("log order stats density X : ",X_candidate,X_candidate)
    #             log_density_current = self.log_order_stats_density(X_current, I_i)
                
    #             log_density_candidate = self.log_order_stats_density(X_candidate, I_i)
                
    #             ratio = np.exp(log_density_candidate - log_density_current)
            
    #             #print("Q{} : current = {} (llike = {}) candidate = {} (llike = {}) ratio = {}".format(k,Q_sim_star[i],log_density_current,Q_tot_star2[j],log_density_candidate,ratio))
    #             if np.random.uniform(0, 1) < ratio:
    #                 Q_tot[j - 1] = Q_tot_star2[j]
    #                 Q_tot[j] = Q_tot_star2[j + 1]
    #                 Q_sim[i] = Q_tot_star2[j]
    #             j += 2
    #             i += 1
    #         else:
    #             j += 1
    #         k += 1
    #         #print()
    #     return Q_sim, Q_tot

    # def Resample_X_Q2(self, Q_obs, Q_sim, Q_tot, N, K, I, G, std_prop):
    #     if len(Q_sim) > 0:
    #         Q_sim, Q_tot = self.OrderStats_MH(
    #             Q_obs,
    #             Q_sim,
    #             Q_tot,
    #             N,
    #             K,
    #             I,
    #             G,
    #             std_prop,
                
    #         )

    #     K1 = [K[0] - 1] + list(K[1:] - K[:-1] - 1) + [N - K[-1]]
    #     X1 = np.insert(np.array(Q_tot).astype(float), 0, -np.inf)
    #     X2 = np.append(Q_tot, np.inf)
    #     a, b = np.repeat(X1, K1), np.repeat(X2, K1)
    #     sample = self._distribution.truncated(a=a, b=b,size=len(a))
    #     return np.round(np.append(sample, Q_tot).reshape(-1), 8), Q_sim, Q_tot
    
    def posterior(self,X, std_prop):
        """Function to sample from the posterior of parameters theta given data X."""
        
        current_theta = self.parameters_value
        
        if self.type_distribution.name=="Normal" and self.parameters_dict["loc"].name == "normal" and self.parameters_dict["scale"].name == "inv_gamma":
            mu_0, sigma_0 = self.parameters_dict["loc"].parameters["loc"].value, self.parameters_dict["scale"].parameters["scale"].value
            alpha,beta = self.parameters_dict["scale"].parameters["shape"],self.parameters_dict["scale"].parameters["scale"]
            
            

        #print("In posterior : current_theta = {} Standard prop = {}".format(current_theta,std_prop))
        for param_name, param in self.parameters_dict.items():
            current_value = self.parameters_value[param_name]
            proposed_value = np.random.normal(current_value, std_prop[param_name])
            
            if not param._check_domain([proposed_value]):
                print("CONTINUE {}",proposed_value)
                continue
            # Calculer le ratio de Metropolis-Hastings
            #if param_name=='loc':
                #print("MIN X = {} current loc = {} proposed loc = {}".format(np.min(X),current_theta['loc'],proposed_value))
            current_llikelihood = self.type_distribution(theta=list(current_theta.values())).llikelihood(X)
            proposed_theta = current_theta.copy()
            proposed_theta[param_name]=proposed_value
            proposed_llikelihood = self.type_distribution(theta=list(proposed_theta.values())).llikelihood(X)
            
            current_lprior = param._distribution.logpdf(current_value)
            proposed_lprior = param._distribution.logpdf(proposed_value)
            
            ratio = np.exp(proposed_llikelihood - current_llikelihood + proposed_lprior - current_lprior)
            # if param_name=='loc':
            #     print("Min(X) = {}".format(np.min(X)))
            #     print("Current = {} llike = {:.2E} lprior = {:.2E}".format(current_value,current_llikelihood,current_lprior))
            #     print("Proposed = {} llike = {:.2E} lprior = {:.2E}".format(proposed_value,proposed_llikelihood,proposed_lprior))
            #     print("Diff llike = {:.2E} lprior = {:.2E} total = {:.2E} ratio = {:.2E}\n\n".format(proposed_llikelihood-current_llikelihood,proposed_lprior-current_lprior,proposed_llikelihood-current_llikelihood+proposed_lprior-current_lprior,ratio))
            #print('{}:\nproposed = {:.2E} llikelihood = {:.2E} lprior = {:.2E}\ncurrent = {:.2E} llikelihood = {:.2E} lprior = {:.2E}\ndiff llike = {} lprior = {} total = {} ratio = {}'.format(param_name,proposed_value,proposed_llikelihood,proposed_lprior,current_value,current_llikelihood,current_lprior, proposed_llikelihood-current_llikelihood,proposed_lprior-current_lprior,proposed_llikelihood-current_llikelihood+proposed_lprior-current_lprior,ratio))
            if np.random.uniform(0, 1) < ratio:
                current_theta[param_name]=proposed_value
            #print("{}: current theta = {} proposed theta = {} proposed_llikelihood = {} current_llikelihood = {} min(X) = {}".format(param_name,list(current_theta.values()),list(proposed_theta.values()),proposed_llikelihood,current_llikelihood,np.min(X)))
        #print(current_theta)
        self._distribution= self.type_distribution(theta=list(current_theta.values()))
        return current_theta

    def Gibbs_Quantile(self,T: int,N: int,Q: list,P: list, std_prop_dict: dict = {}, std_prop_quantile=0.1, List_X=False, epsilon=0.001, verbose=True, True_X=[]):
                        
        """
        Gibbs sampler to sample from the posterior of model parameters given a sequence of quantiles.

        Args:
            T (int): Number of iterations.
            N (int): Size of the vector X.
            Q (list): Observed quantile value.
            P (float): Probability associated to the observed quantiles Q.
            distribution (str): Distribution of the data ("normal", "cauchy", "weibull", or "TranslatedWeibull").
            prior_loc (str): Prior distribution of the location parameter ("normal", "cauchy", "uniform", or "none").
            prior_scale (str): Prior distribution of the scale parameter ("gamma","jeffreys").
            prior_shape (str): Prior distribution of the shape parameter ("gamma").
            par_prior_loc (list, optional): Prior hyperparameters for the location parameter. Defaults to [0, 1].
            par_prior_scale (list, optional): Prior hyperparameters for the scale parameter. Defaults to [1, 1].
            par_prior_shape (list, optional): Prior hyperparameters for the shape parameter. Defaults to [0, 1].
            std_prop_loc (float, optional): Standard deviation of the RWMH Kernel for the location parameter. Defaults to 0.1.
            std_prop_scale (float, optional): Standard deviation of the RWMH Kernel for the scale parameter. Defaults to 0.1.
            std_prop_shape (float, optional): Standard deviation of the RWMH Kernel for the shape parameter. Defaults to 0.1.
            List_X (bool, optional): If True, will return the list of all latent vectors X. Otherwise, it will return the first and the last. Defaults to False.
            verbose (bool, optional): If True, will display the progression of the sampling. Defaults to True.
        Returns:
            A dictionary containing:
                chains (dict): The chains sampled from the parameters' posterior.
                X (list): List of latent vectors.
                Q_sim (list): List of all simulated order statistics at each iteration.
                Q_tot (list): List of all order statistics considered at each iteration.
                 input parameters"""
                 
                 
        print("Init...")
        par_names = list(self.parameters_dict.keys())
        if std_prop_dict == {}:
            std_prop_dict = {param_name: 0.1 for param_name in par_names}
        X, q, q_tot, q_sim, I, I_sim, G = self.Quantile_Init(
            Q, P, N, epsilon=epsilon)
        Chains = {par_name: [] for par_name in par_names}
        
        X_list = [X]
        Q_Tot = [q_tot]
        Q_Sim = [q_sim]
        Llike = [self._distribution.llikelihood(X)]
        print("Init done!",self.parameters_value)
        for i in tqdm(range(T), disable=not (verbose)):
            #print("GO for Sampling X and Q_sim...")
            #print("Resampling of X...")
            #print("In Gibbs_Quantile : Iteration {} Reparametrization ≠ {}".format(i,reparametrization))
            if True_X==[]:
                X,q_tot,q_sim = self.Resample_X_Q(
                q,
                q_tot,
                q_sim,
                N,
                I,
                G,
                std_prop_quantile,
            )
            else:
                X=True_X

            theta = self.posterior(X,std_prop_dict)
            #print("Posterior theta = {}\n\n".format(theta))
            for par_name in par_names:
                Chains[par_name].append(theta[par_name])

            Q_Tot.append(list(q_tot))
            Q_Sim.append(list(q_sim))
            if List_X:
                X_list.append(X)
            Llike.append(self._distribution.llikelihood(X))
        if not (List_X):
            X_list.append(X)
        print(Q_Sim)
        if verbose:
            Q = np.array(Q_Sim).T
            print("Q.shape=", Q.shape)
            for i in range(Q.shape[0]):
                q = Q[i]
                print(
                    "Acceptance rate of the order statistics ({}) = {:.2%}".format(
                        int(I_sim[i]), (len(np.unique(Q[i])) - 1) / Q.shape[1]
                    )
                )
        if verbose :
            acceptation_rate = [
                (len(np.unique(Chains[par_name])) - 1) / T for par_name in par_names
            ]
            print("Acceptation rates MH :", end=" ")
            for i in range(len(par_names)):
                print("{} = {:.2%}".format(par_names[i], acceptation_rate[i]), end=" ")
            print()
            
        Q_Tot = np.array([np.hstack(q_tot).reshape(-1) for q_tot in Q_Tot])
        return {
            "X": X_list,
            "chains": Chains,
            "N": N,
            "Q": Q,
            "P": P,
            "Q_sim": np.array(Q_Sim),
            "Q_tot": np.array(Q_Tot),
            "T": T,
            "Llike": Llike,
        }
    ### MAD FUNCTIONS
    
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
                    a,b = med2, med
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
                a1, a2, b1, b2 = -np.inf, med1, med2, np.inf
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
                    # if distribution == "normal":
                    #     xnew1 = norm(loc=loc, scale=scale).rvs(1)[0]
                    # elif distribution == "cauchy":
                    #     xnew1 = cauchy(loc=loc, scale=scale).rvs(1)[0]
                    # elif distribution == "weibull":
                    #     xnew1 = weibull_min(c=shape, scale=scale, loc=loc).rvs(1)[0]
                    # elif distribution == "translated_weibull":
                    #     xnew1 = weibull_min(c=shape, scale=scale).rvs(1)[0]
                    # elif distribution == "lognormal" or distribution == "translated_lognormal":
                    #     xnew1 = lognorm(loc = loc, c=shape, scale=scale).rvs(1)[0]
                    # elif distribution == "generalized_pareto":
                    #     xnew1 = genpareto(c=shape, scale=scale, loc=loc).rvs(1)[0]
                        
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
        
    
    def Gibbs_med_MAD(self,T: int,N: int,med: float,MAD: float, std_prop_dict: dict = {}, List_X=False, epsilon=0.001, verbose=True, True_X=[]):
        #print("Init...")
        par_names = list(self.parameters_dict.keys())
        if std_prop_dict == {}:
            std_prop_dict = {param_name: 0.1 for param_name in par_names}
        X = self.med_MAD_Init(med,MAD,N)
        #print("Init done!",self.parameters_value)
        Chains = {par_name: [] for par_name in par_names}
        
        X_list = [X]

        Llike = [self._distribution.llikelihood(X)]
        
        for i in tqdm(range(T), disable=not (verbose)):
            if True_X==[]: X = self.Resample_X_med_MAD(X,med,MAD)
            else: X=True_X
            theta = self.posterior(X,std_prop_dict)
            for par_name in par_names:
                Chains[par_name].append(theta[par_name])
                
            if List_X: X_list.append(X)
            Llike.append(self._distribution.llikelihood(X))
        if not (List_X):
            X_list.append(X)

      
        if verbose :
            acceptation_rate = [
                (len(np.unique(Chains[par_name])) - 1) / T for par_name in par_names
            ]
            print("Acceptation rates MH :", end=" ")
            for i in range(len(par_names)):
                print("{} = {:.2%}".format(par_names[i], acceptation_rate[i]), end=" ")
            print()
        return {
            "X": X_list,
            "chains": Chains,
            "N": N,
            "T": T,
            "Llike": Llike,
        }
    


class NormalModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.type_distribution = Normal
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (norm(loc).ppf(P[-1]) - norm(loc).ppf(P[0]))
        self.parameters_value = {'loc': loc, 'scale': scale}
        self._distribution = Normal(loc=loc, scale=scale)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G 
 
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        scale = MAD*1.4826
        self.parameters_value = {'loc': loc, 'scale': scale}
        self._distribution = Normal(loc=loc, scale=scale)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
    
class NormalKnownScaleModel(Model):
    def __init__(self, loc:Distribution) -> None:
        self.loc = loc
        self.type_distribution = Normal_known_scale
        self.parameters_dict = {'loc': self.loc}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        self.parameters_value = {'loc': loc}
        self._distribution = Normal_known_scale(loc=loc)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
        
class CauchyModel(Model):
    
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.type_distribution = Cauchy
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (cauchy(loc).ppf(P[-1]) - cauchy(loc).ppf(P[0]))
        self.parameters_value = {'loc': loc, 'scale': scale}
        self._distribution = Cauchy(loc=loc, scale=scale)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G       


class GammaModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        self.scale = scale
        self.shape = shape
        self.type_distribution = Gamma 
        self.parameters_dict = {'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (
            gamma(shape).ppf(P[-1])
            - gamma(shape).ppf(P[0])
        )
        self.parameters_value = {'scale': scale, 'shape': shape}
        self._distribution = Gamma(scale=scale, shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G

class ReparametrizedGammaModel(Model):
    
    def __init__(self, mean:Distribution, std:Distribution) -> None:
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedGamma 
        self.parameters_dict = {'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)  
    
    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):

        scale = (Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (gamma(scale=scale, a=2).ppf(P[-1]) - gamma(scale=scale, a=2).ppf(P[0]))
        mean = scale*shape
        std = np.sqrt(scale**2*shape)
        self.parameters_value = {'mean': mean, 'std': std}
        self._distribution = ReparametrizedGamma(mean=mean,std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

class TranslatedGammaModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedGamma 
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))

    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (gamma(loc=loc,scale=scale).ppf(P[-1]) - gamma(loc=loc,scale=scale).ppf(P[0]))
        self.parameters_value = {'loc': loc, 'scale': scale, 'shape': shape}
        self._distribution = TranslatedGamma(loc=loc,scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    


class ReparametrizedTranslatedGammaModel(Model):
    
    def __init__(self, loc:Distribution, mean:Distribution, std:Distribution) -> None:
        self.loc = loc
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedTranslatedGamma 
        self.parameters_dict = {'loc': self.loc, 'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon):
        loc = 2*Q[0]-Q[1]
        scale = (Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (gamma(a=2,loc=loc,scale=scale).ppf(P[-1]) - gamma(a=2,loc=loc,scale=scale).ppf(P[0]))
        mean = scale*shape+loc
        std = np.sqrt(shape)*scale
        self.parameters_value = {'loc': loc, 'mean': mean, 'std': std}
        self._distribution = ReparametrizedTranslatedGamma(loc=loc,mean=mean,std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class FixedReparametrizedTranslatedGammaModel(Model):
    
    def __init__(self,mean:Distribution, std:Distribution) -> None:

        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedTranslatedGamma 
        self.parameters_dict = {'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (6389.64, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon):
        loc = 6389.64
        scale = (Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (gamma(a=2,loc=loc,scale=scale).ppf(P[-1]) - gamma(a=2,loc=loc,scale=scale).ppf(P[0]))
        mean = scale*shape+loc
        std = np.sqrt(shape)*scale
        self.parameters_value = {'mean': mean, 'std': std}
        self._distribution = FixedReparametrizedTranslatedGamma(mean=mean,std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

class LogNormalModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        self.scale = scale
        self.shape = shape
        self.type_distribution = LogNormal 
        self.parameters_dict = {'scale': self.scale, 'shape': self.shape}
        self.parameters_value = {'scale': None, 'shape': None}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (lognorm(s=1, scale=np.exp(scale)).ppf(P[-1]) - lognorm(s=1, scale=np.exp(scale)).ppf(P[0]))
        
        self.parameters_value = {'scale': scale, 'shape': shape}
        self._distribution = LogNormal(scale=scale, shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class ReparametrizedLogNormalModel(Model):
    
        def __init__(self, mean:Distribution, std:Distribution) -> None:
            self.mean = mean
            self.std = std
            self.type_distribution = ReparametrizedLogNormal 
            self.parameters_dict = {'mean': self.mean, 'std': self.std}
            self.parameters_value = {'mean': None, 'std': None}
            super().__init__(self.parameters_dict)
            
        def domain(self) -> Tuple[float, float]:
            return (0, float('inf'))
        
        def Quantile_Init(self, Q, P, N, epsilon=0.001):
            scale = np.log(Q[len(Q) // 2])
            shape = (Q[-1] - Q[0]) / (lognorm(s=1, scale=np.exp(scale)).ppf(P[-1]) - lognorm(s=1, scale=np.exp(scale)).ppf(P[0]))
            mean = np.exp(scale+shape**2/2)
            std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*scale+shape**2))
            self.parameters_value = {'mean': mean, 'std': std}
            self._distribution = ReparametrizedLogNormal(mean=mean,std=std)
            X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
            return X_0, Q, Q_tot, Q_sim, I, I_sim, G

class TranslatedLogNormalModel(Model):
    
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedLogNormal 
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale, 'shape': self.shape}
        self.parameters_value = {'loc': None, 'scale': None, 'shape': None}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[-1]) - lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[0]))
    
        self.parameters_value = {'loc': loc, 'scale': scale, 'shape': shape}
        print(self.parameters_value)
        self._distribution = TranslatedLogNormal(loc=loc,scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class ReparametrizedTranslatedLogNormalModel(Model):
    
    def __init__(self, loc:Distribution, mean:Distribution, std:Distribution) -> None:
        self.loc = loc
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedTranslatedLogNormal 
        self.parameters_dict = {'loc': self.loc, 'mean': self.mean, 'std': self.std}
        self.parameters_value = {'loc': None, 'mean': None, 'std': None}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[-1]) - lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[0]))
        
        mean = np.exp(scale+shape**2/2)+loc
        std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*scale+shape**2))
        self.parameters_value = {'loc': loc, 'mean': mean, 'std': std}
        self._distribution = ReparametrizedTranslatedLogNormal(loc=loc,mean=mean,std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

class WeibullModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        self.scale = scale
        self.shape = shape
        self.type_distribution = Weibull #to access corresponding distribution in fit
        self.parameters_dict = {'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))

    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (
            weibull_min(shape).ppf(P[-1])
            - weibull_min(shape).ppf(P[0])
        )
        self._distribution = TranslatedWeibull(scale=scale,shape=shape)
        self.parameters_value = {'scale': scale, 'shape': shape}
        #print("Init_X_Quantile...")
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        #print("Quantile init done!")
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G 
    
class TranslatedWeibullModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedWeibull #to access corresponding distribution in fit
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N,init_theta=[], epsilon=0.001):
        if init_theta==[]:
            loc = 2*Q[0]-Q[1]
            shape = 1.5
            scale = (Q[-1] - Q[0]) / (
                weibull_min(shape, loc=loc).ppf(P[-1])
                - weibull_min(shape, loc=loc).ppf(P[0])
            )
        else: loc,scale,shape=init_theta
        self.parameters_value = {'loc': loc, 'scale': scale, 'shape': shape}
        self._distribution = TranslatedWeibull(loc=loc, scale=scale,shape=shape)
        #print("Init_X_Quantile...")
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        #print("Quantile init done!")
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G 
    
class GeneralizedParetoModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = GeneralizedPareto
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        shape = .1
        scale = (Q[-1] - Q[0]) / (self.type_distribution(loc=loc, shape=shape)._distribution.ppf(P[-1]) -  self.type_distribution(loc=loc, shape=shape)._distribution.ppf(P[0]))
        
        self.parameters_value = {'loc': loc, 'scale': scale, 'shape': shape}
        self._distribution = GeneralizedPareto(loc=loc, scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G

class ReparametrizedGeneralizedParetoModel(Model):
    
    def __init__(self, loc:Distribution, mean:Distribution, std:Distribution) -> None:
        self.loc = loc
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedGeneralizedPareto
        self.parameters_dict = {'loc': self.loc, 'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        shape = .1
        scale = (Q[-1] - Q[0]) / (genpareto(loc=loc, c=shape).ppf(P[-1]) -  genpareto(loc=loc, c=shape).ppf(P[0]))
        
        mean = loc+scale/(1-shape)
        std = np.sqrt(scale**2/(1-shape)**2/(1-2*shape))
        self.parameters_value = {'loc': loc, 'mean': mean, 'std': std}
        self._distribution = ReparametrizedGeneralizedPareto(loc=loc,mean=mean,std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class ParetoType2Model(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = ParetoType2
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        shape = 2.5
        scale = (Q[-1] - Q[0]) / (self.type_distribution(loc=loc, shape=shape)._distribution.ppf(P[-1]) - self.type_distribution(loc=loc, shape=shape)._distribution.ppf(P[0]))
        self.parameters_value = {'loc': loc, 'scale': scale, 'shape': shape}
        self._distribution = ParetoType2(loc=loc, scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class ReparametrizedParetoType2Model(Model):
    def __init__(self, loc:Distribution, mean:Distribution, std:Distribution) -> None:
        self.loc = loc
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedParetoType2
        self.parameters_dict = {'loc': self.loc, 'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        shape = 2.5
        scale = (Q[-1] - Q[0]) /  (ParetoType2(loc=loc, shape=shape)._distribution.ppf(P[-1]) - ParetoType2(loc=loc, shape=shape)._distribution.ppf(P[0]))
        
        mean = loc+scale/(shape-1)
        std = scale/(shape-1)*np.sqrt(shape/(shape-2))
        self.parameters_value = {'loc': loc, 'mean': mean, 'std': std}
        self._distribution = ReparametrizedParetoType2(loc=loc,mean=mean,std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
        
class FixedTranslatedLogNormalModel(Model):
    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        self.scale = scale
        self.shape = shape
        self.type_distribution = FixedTranslatedLogNormal
        self.parameters_dict = {'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (6389.64, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 6389.64
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[-1]) - lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[0]))
    
        self.parameters_value = {'scale': scale, 'shape': shape}
        self._distribution = TranslatedLogNormal(loc=loc,scale=scale,shape=shape)
        print(self.parameters_value)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

class FixedGeneralizedPareto(Model):
    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        self.scale = scale
        self.shape = shape
        self.type_distribution = FixedGeneralizedPareto
        self.parameters_dict = {'scale': self.scale, 'shape': self.shape}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (6389.64, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 6389.64
        shape = .1
        scale = (Q[-1] - Q[0]) / (genpareto(loc=loc, c=shape).ppf(P[-1]) -  genpareto(loc=loc, c=shape).ppf(P[0]))
        
        self.parameters_value = {'scale': scale, 'shape': shape}
        self._distribution = GeneralizedPareto(loc=loc, scale=scale,shape=shape)
        print(self.parameters_value)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

class FixedReparametrizedTranslatedLogNormalModel(Model):
    def __init__(self, mean:Distribution, std:Distribution) -> None:
        self.mean = mean
        self.std = std
        self.type_distribution = FixedReparametrizedTranslatedLogNormal
        self.parameters_dict = {'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (6389.64, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 6389.64
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[-1]) - lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[0]))
        
        mean = np.exp(scale+shape**2/2)+loc
        std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*scale+shape**2))
        mean,std = 25000, 10000
        self.parameters_value = {'mean': mean, 'std': std}
        self._distribution = ReparametrizedTranslatedLogNormal(loc=loc,mean=mean,std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class FixedReparametrizedGeneralizedParetoModel(Model):
    def __init__(self, mean:Distribution, std:Distribution) -> None:
        self.mean = mean
        self.std = std
        self.type_distribution = FixedReparametrizedGeneralizedPareto
        self.parameters_dict = {'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (6389.64, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 6389.64
        shape = .1
        scale = (Q[-1] - Q[0]) / (genpareto(loc=loc, c=shape).ppf(P[-1]) -  genpareto(loc=loc, c=shape).ppf(P[0]))
        
        mean = loc+scale/(1-shape)
        std = scale/(1-shape)/np.sqrt(1-2*shape)
        mean,std = 25000, 20000
        self.parameters_value = {'mean': mean, 'std': std}
        self._distribution = ReparametrizedGeneralizedPareto(loc=loc,mean=mean,std=std)
        print(self.parameters_value)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class FixedReparametrizedParetoType2Model(Model):
    def __init__(self, mean:Distribution, std:Distribution) -> None:
        self.mean = mean
        self.std = std
        self.type_distribution = FixedReparametrizedParetoType2
        self.parameters_dict = {'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (6389.64, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 6389.64
        shape = 2.5
        scale = (Q[-1] - Q[0]) /  (ParetoType2(loc=loc, shape=shape)._distribution.ppf(P[-1]) - ParetoType2(loc=loc, shape=shape)._distribution.ppf(P[0]))
        
        mean = loc+scale/(shape-1)
        std = scale/(shape-1)*np.sqrt(shape/(shape-2))
        mean,std=17000, 11000
        print(mean,std)
        #mean,std = 24000, 16000
        self.parameters_value = {'mean': mean, 'std': std}
        self._distribution = ReparametrizedParetoType2(loc=loc,mean=mean,std=std)
        print(self.parameters_value)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class LaplaceModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.type_distribution = Laplace
        self.parameters_dict = {'loc': self.loc, 'scale': self.scale}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (laplace(loc).ppf(P[-1]) - laplace(loc).ppf(P[0]))
        self.parameters_value = {'loc': loc, 'scale': scale}
        self._distribution = Laplace(loc=loc, scale=scale)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon) 
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        scale = MAD/np.log(2)
        self.parameters_value = {'loc': loc, 'scale': scale}
        self._distribution = Laplace(loc=loc, scale=scale)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0

class LaplaceKnownScaleModel(Model):
    def __init__(self, loc:Distribution) -> None:
        self.loc = loc
        self.type_distribution = Laplace_known_scale
        self.parameters_dict = {'loc': self.loc}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        self.parameters_value = {'loc': loc}
        self._distribution = Laplace_known_scale(loc=loc)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
        
class ReparametrizedLaplaceModel(Model):
    def __init__(self, mean:Distribution, std:Distribution) -> None:
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedLaplace
        self.parameters_dict = {'mean': self.mean, 'std': self.std}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))

    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (laplace(loc).ppf(P[-1]) - laplace(loc).ppf(P[0]))
        std = scale*np.sqrt(2)
        mean = loc
        self.parameters_value = {'mean': mean, 'std': std}
        self._distribution = ReparametrizedLaplace(mean=mean, std=std)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        scale = MAD/np.log(2)
        self.parameters_value = {'mean': loc, 'std': scale*np.sqrt(2)}
        self._distribution = ReparametrizedLaplace(mean=loc, std=scale*np.sqrt(2))
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
    