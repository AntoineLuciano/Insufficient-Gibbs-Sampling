from scipy.stats import nbinom,poisson,beta,expon,gamma,betaprime,uniform, cauchy, norm, multinomial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.integrate import quad
## Theoric
from scipy.special import factorial, gammaln,comb, betaln

def combln(n, k):
    return gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
def logBF_theo_y(y):
    N = len(y)
    S = np.sum(y)
    return gammaln(N+S+2)-(S+1)*np.log(N+1)-np.sum(gammaln(y+1))-gammaln(N+1)

def logBF_theo_S(S,N):
    return (S-1)*np.log(N)+np.log(N+S)+np.log(N+S+1)-(S+1)*np.log(N+1)
# def BF_theo_pois_nb(y,r=1):
#     N = len(y)
#     S = np.sum(y)
#     if r!=1: 
#         return factorial(S)/((N+1)**(S+1)*np.prod(factorial(y))*np.prod(comb(y+r-1,y))*beta(1+N*r-S,S+1))
#     else:
#         return factorial(S)/(N+1)**(S+1)/np.prod(factorial(y))/beta(1+N,S+1)
    
def logBF_theo_pois_nb(y,r=1):
    N=len(y)
    S=np.sum(y)
    if r!=1:
        return gammaln(S+1)-(S+1)*np.log(N+1)-np.sum(gammaln(y+1))-np.sum(gammaln(y+r)-gammaln(y+1))-gammaln(S+1)-betaln(1+N*r-S,S+1)
    else:
        return gammaln(S+1)-(S+1)*np.log(N+1)-np.sum(gammaln(y+1))-betaln(1+N,S+1)
# def BF_S_theo_pois_nb(S,N,r=1):
#     if r!=1:
#         return N**S/((N+1)**(S+1)*beta(1+N*r,S+1)*comb(N*r+S-1,S))
#     else:
#         return N**S/((N+1)**(S+1)*beta(1+N,S+1)*comb(N+S-1,S))
    
def logBF_S_theo_pois_nb(S,N,r=1):
    if r!=1:
        return S*np.log(N)-(S+1)*np.log(N+1)-betaln(1+N*r,S+1)-np.log(comb(N*r+S-1,S))
    else:
        return S*np.log(N)-(S+1)*np.log(N+1)-betaln(N+1,S+1)-combln(N+S-1,S)
def logBF_S(S,N):
    I_1 = quad(lambda mu: np.exp(-mu*(N+1))*mu**S, 0, np.inf)[0]
    I_2 = quad(lambda p: p**N*(1-p)**(S), 0, 1)[0]
    c_1 = N**S/factorial(S)
    c_2 = comb(N+S-1,S)
    
    return I_1*c_1/I_2/c_2
## Bridge

from scipy.optimize import fixed_point

def func_r_paper(r,l1,l2,lstar):
    lstar = np.median(l1)
    return np.sum(np.exp(l2-lstar)/(np.exp(l2-lstar)+r))/np.sum(1/(np.exp(l1-lstar)+r))

def fixedpoint(func,x0,args,eps=1e-4,max_iter=100000):
    res = [np.inf,x0]
    while np.abs(res[-1]-res[-2])>eps and len(res)<max_iter:
        res.append(func(res[-1],*args)) 
    return res
        
def BF_paper(l1,l2):
    lstar = np.median(l1)
    r = fixed_point(func_r_paper,np.exp(-lstar),args=(l1,l2,lstar))
    BF = r*np.exp(lstar)
    return BF
    
def BF_paper_test(l1,l2):
    lstar = np.median(l1)
    r = fixedpoint(func_r_paper,np.exp(-lstar),args=(l1,l2,lstar))
    BF = np.array(r)*np.exp(lstar)
    return BF



def logratio_pois_nb(mu,y,r=1):
    S = np.sum(y)
    N = len(y)
    return -mu*(N+1)-np.sum(gammaln(y+1))+(N+S+2)*np.log(1+mu)

def logratio_pois_nb(mu,y,r=1):
    S = np.sum(y)
    N = len(y)
    return np.sum(gamma(S+1,1/(1+mu)).logpdf(y))-np.sum(nbinom(S+1,1/(1+mu)).logpmf(y))

def sample_X_poiss_given_S(S,N,T):
    return multinomial.rvs(n=S, p=[1/N]*N,size=T)


def sample_X_nbinom_given_S(S,N,T):
    res=[]
    for _ in tqdm(range(T)):
        vect = nbinom(p = N/(N+S), n=1).rvs(N)
        while np.sum(vect)!=S:
            vect = nbinom(p = N/(N+S), n=1).rvs(N)
        res.append(vect)
    return np.array(res).squeeze()

def sample_nbinom(S,N):
    vect = np.zeros(N+S-1)
    index = np.random.choice(N+S-1,N-1, replace=False)
    vect[index] = 1
    res = []
    i=0
    count = 0
    for i in range(N+S-1):
        if vect[i]==0:
            count+=1
        else:
            res.append(count)
            count = 0
    res.append(count)
        

    return np.array(res)

def sample_X_nbinom_given_S(S,N,T):
    return np.array([sample_nbinom(S,N) for _ in range(T)]) 

def post_nbinom_unif(S,N,size):
    return betaprime(S+1,N+1).rvs(size)

def post_pois_exp(S,N,size):
    return gamma(S+1,scale=1/(N+1)).rvs(size)

def func_BF(i,S,N):
    X_poiss_S = sample_X_poiss_given_S(S,N,T)
    X_nb_S = sample_X_nbinom_given_S(S,N,T)
    mu_poiss = post_pois_exp(S,N,T)
    mu_nb = post_nbinom_unif(S,N,T)
    l1 = np.array([logratio_pois_nb(mu,X) for mu,X in zip(mu_nb,X_nb_S)])
    l2 = np.array([logratio_pois_nb(mu,X) for mu,X in zip(mu_poiss,X_poiss_S)])
    Bridge = BF_paper(l1,l2)
    return Bridge

from multiprocessing import Pool, cpu_count
import csv
import numpy as np
print("Number of processors: ", cpu_count())
print("\n\n ----- N = 50 ----- \n\n")
N = 50 
S_list = np.arange(0,4*N+1,(4*N)/10).astype(int)
n_iter = 100
T = 100000
Theo_list, Bridge_list = [],[]
for S in tqdm(S_list):
    Bridge_S =[]
    Theo = np.exp(logBF_theo_S(S,N))
    pool = Pool(cpu_count())
    Bridge_list.append(list(pool.starmap(func_BF, [(i,S,N) for i in tqdm(range(n_iter))])))
    pool.close()
    pool.join()    
    print("S = {} Theoretical logBF = {} Bridge logBF = {}".format(S,Theo,np.mean(Bridge_S)))
    Theo_list.append(Theo)
    
# Create a list of S values for each column
S_values = S_list.tolist()

# Transpose the Bridge_list to have S values as columns
Bridge_list_transposed = np.transpose(Bridge_list)

# Define the file path
csv_file = './bridge_list.csv'

# Write the Bridge_list_transposed to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(S_values)  # Write the S values as the first row
    writer.writerows(Bridge_list_transposed)

# N = 100
print("\n\n ----- N = 100 ----- \n\n")
N = 100
S_list = np.arange(0,4*N+1,(4*N)/10).astype(int)
T = 100000
n_iter = 100
Theo_list2, Bridge_list2 = [],[]
logBridge2 = []
for S in tqdm(S_list):
    Theo = np.exp(logBF_theo_S(S,N))
    Bridge_list2.append(list(pool.starmap(func_BF, [(i,S,N) for i in tqdm(range(n_iter))])))
    pool.close()
    pool.join()   
    print("S = {} Theoretical logBF = {} Bridge logBF = {}".format(S,Theo,np.mean(Bridge_S)))
    Theo_list2.append(Theo)
    

S_values2 = S_list.tolist()

# Transpose the Bridge_list to have S values as columns
Bridge_list_transposed2 = np.transpose(Bridge_list)

# Define the file path
csv_file = './bridge_list2.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(S_values2)  # Write the S values as the first row
    writer.writerows(Bridge_list_transposed2)

## N = 1000
N = 1000
S_list = np.arange(0,4*N+1,(4*N)/10).astype(int)
T = 100000
n_iter = 10
Theo_list3, Bridge_list3 = [],[]
logBridge3 = []
for S in tqdm(S_list):
    
    Theo = np.exp(logBF_theo_S(S,N))
    Bridge_list3.append(list(pool.starmap(func_BF, [(i,S,N) for i in tqdm(range(n_iter))])))
    pool.close()
    pool.join()   
    print("S = {} Theoretical logBF = {} Bridge logBF = {}".format(S,Theo,np.mean(Bridge_S)))
    Theo_list3.append(Theo)
    pool.close()
    
S_values3 = S_list.tolist()

# Transpose the Bridge_list to have S values as columns
Bridge_list_transposed3 = np.transpose(Bridge_list)

# Define the file path
csv_file = './bridge_list3.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(S_values3)  # Write the S values as the first row
    writer.writerows(Bridge_list_transposed3)
