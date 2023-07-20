import numpy as np 
import scipy
from even_final import * 
from odd_final import * 
from truncated import *
from normal_post import *
from tqdm import tqdm
from cauchy_post import *
from weibull_post import *


def move2even(
    X,
    loc=None,
    scale=None,
    verbose=False,
    index=None,
    med=None,
    MAD=None,
    par=[],
    distribution="normal",
    shape=1
):
    
    if index == None:
        index = np.random.choice(len(X), 2, replace=False)
    X = np.array(X)
    xij = X[index]
    xi, xj = xij[0], xij[1]
    n = len(X) // 2
    if len(par) == 0:
        n = len(X) // 2
        X_s = np.sort(X)
        med1 = X_s[n - 1]
        med2 = X_s[n]
        S = np.abs(X - med)
        S_s = np.sort(S)
        MAD1, MAD2 = S_s[n - 1], S_s[n]
        [i_MAD1, i_MAD2] = np.argsort(S)[n - 1 : n + 1]
        Xmad1, Xmad2 = X[[i_MAD1, i_MAD2]]
        par = [MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2]

    par = np.round(par, 10)

    [MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2] = par

    change_meded = False
    change_MAD = False

    if sorted(xij) == [med1, med2]:
        case = "1"
        s3 = np.sort(np.abs(X - med))[2]

        a, b = med - s3, med + s3
        xnew1 = truncated(
            a=(a - loc) / scale,
            b=(b - loc) / scale,
            size=1,
            loc=loc,
            scale=scale,
            distribution=distribution, shape=shape
        )[0]
    
        xnew2 = sym(med, xnew1)

        change_meded = True
    elif sorted(xij) == sorted([Xmad1, Xmad2]):
        S = np.sort(np.abs(X - med))
        epsilon = np.minimum(MAD1 - S[n - 2], S[n + 1] - MAD2)
        if xi < med and xj < med:
            case = "2b"
            a, b = med - MAD2 - epsilon, med - MAD1 + epsilon
            if a >= b:
                print("BIZARRE 2B")
                xnew1, xnew2 = xi, xj
            else:
                xnew1 = truncated(
                    a=(a - loc) / scale,
                    b=(b - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
                xnew2 = sym(med - MAD, xnew1)

        elif xi > med and xj > med:
            case = "2a"
            a, b = med + MAD1 - epsilon, med + MAD2 + epsilon
            if a >= b:
                xnew1, xnew2 = xi, xj
                print("BIZARRE 2A")
            else:
                xnew1 = truncated(
                    a=(a - loc) / scale,
                    b=(b - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
                xnew2 = sym(med + MAD, xnew1)
        else:
            case = "2c"
            a1, b1, a2, b2 = (
                med - MAD2 - epsilon,
                med - MAD1 + epsilon,
                med + MAD1 - epsilon,
                med + MAD2 + epsilon,
            )
            if a1 == b1 or a2 == b2:
                print("BIZARRE 2C")
                xnew1, xnew2 = xi, xj
            else:
                xnew1 = truncated_2inter(
                    loc, scale, a1, b1, a2, b2, distribution=distribution,shape=shape
                )[0]
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
            print("probleme 4")
        if xm == med1 and med + MAD1 > xother > med:
            case = "4a"
            a, b = med, med + MAD1
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
            if xnew1 < med2:
                xnew2 = sym(med, xnew1)
                change_med = True
            else:
                xnew2 = xm
        elif xm == med2 and med - MAD1 < xother < med:
            case = "4b"
            a, b = med - MAD1, med
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
            if xnew1 > med1:
                xnew2 = sym(med, xnew1)
                change_med = True
            else:
                xnew2 = xm
        else:
            case = "4c"
            a, b = zone_even_ab(xother, med1, med2, MAD1, MAD2)
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
            xnew2 = xm
    elif Xmad1 in xij or Xmad2 in xij:
        if xi in [Xmad1, Xmad2]:
            xmad, xother = xi, xj
        elif xj in [Xmad1, Xmad2]:
            xmad, xother = xj, xi
        else:
            print("PROBLEME 5")
        if (xmad - med) * (xother - med) > 0 and (np.abs(xmad - med) - MAD) * (
            np.abs(xother - med) - MAD
        ) > 0:
            case = "5b "
            a, b = zone_even_ab(xother, med1, med2, MAD1, MAD2)
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
            xnew2 = xmad
        elif (xmad - med) * (xother - med) > 0 and (np.abs(xmad - med) - MAD) * (
            np.abs(xother - med) - MAD
        ) < 0:
            case = "5a"
            a, b = zone_even_E_ab(xother, med1, med2, MAD1, MAD2)
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
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
            xnew1 = truncated_2inter(loc, scale, a1, b1, a2, b2,shape=shape)[0]
            if a2 <= xnew1 <= b2:
                xnew2 = sym(med, xmad)
                if xmad == Xmad1:
                    Xmad1 = sym(med, Xmad1)
                elif xmad == Xmad2:
                    Xmad2 = sym(med, Xmad2)
                else:
                    print("\n\PROBLEME 5C\n\n")
            else:
                xnew2 = xmad

        else:
            case = "5d "
            a, b = zone_even_E_ab(xother, med1, med2, MAD1, MAD2)
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
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
            zone_even(xi, X, med=med, MAD=MAD, par=par),
            zone_even(xj, X, med=med, MAD=MAD, par=par),
        ]
        sort_zone = sorted(l_zone)
        if sort_zone in [[1, 2], [3, 4]]:
            case = "6a "
            if xi < med:
                a, b = -np.inf, med1
            else:
                a, b = med2, np.inf
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
            if med - MAD2 <= xnew1 <= med - MAD1:
                xnew2 = sym(med - MAD, xnew1)
                change_MAD = True
            elif med + MAD1 <= xnew1 <= med + MAD2:
                xnew2 = sym(med + MAD, xnew1)
                change_MAD = True
            elif xnew1 < med - MAD2:
                xnew2 = truncated(
                    a=(med - MAD1 - loc) / scale,
                    b=(med1 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
            elif xnew1 > med + MAD2:
                xnew2 = truncated(
                    a=(med2 - loc) / scale,
                    b=(med + MAD1 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
            elif med1 > xnew1 > med - MAD1:
                xnew2 = truncated(
                    a=-np.inf,
                    b=(med - MAD2 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
            else:
                xnew2 = truncated(
                    a=(med + MAD2 - loc) / scale,
                    b=np.inf,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
        elif sort_zone == [2, 3]:
            case = "6b "
            xnew1 = truncated(
                a=(med - MAD1 - loc) / scale,
                b=(med + MAD1 - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
            if med1 < xnew1 < med2:
                xnew2 = sym(med, xnew1)
                change_med = True
            elif xnew1 < med1:
                xnew2 = truncated(
                    a=(med2 - loc) / scale,
                    b=(med + MAD1 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
            else:
                xnew2 = truncated(
                    a=(med - MAD1 - loc) / scale,
                    b=(med1 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
        elif sort_zone in [[1, 3], [2, 4]]:
            case = "6c "
            xnew1 = truncated_2inter(
                loc, scale, -np.inf, med1, med2, np.inf, distribution=distribution
            )[0]
            if med - MAD2 <= xnew1 <= med - MAD1:
                xnew2 = sym(med + MAD, sym(med, xnew1))
                change_MAD = True
            elif med + MAD1 <= xnew1 <= med + MAD2:
                xnew2 = sym(med - MAD, sym(med, xnew1))
                change_MAD = True

            else:
                a, b = zone_even_C_ab(xnew1, med1, med2, MAD1, MAD2)
                xnew2 = truncated(
                    a=(a - loc) / scale,
                    b=(b - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution, shape=shape
                )[0]
        else:
            case = "6d"
            a1, b1 = zone_even_ab(xi, med1, med2, MAD1, MAD2)
            xnew1 = truncated(
                a=(a1 - loc) / scale,
                b=(b1 - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]
            a2, b2 = zone_even_ab(xj, med1, med2, MAD1, MAD2)
            xnew2 = truncated(
                a=(a2 - loc) / scale,
                b=(b2 - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution, shape=shape
            )[0]

    [xnew1, xnew2] = np.round([xnew1, xnew2], 10)
    X[index] = np.array([xnew1, xnew2]).reshape(-1)

    if change_meded:
        [med1, med2] = sorted([xnew1, xnew2])

    if change_MAD:
        #print("Change s")
        S_s = np.sort([np.abs(xnew1 - med), np.abs(xnew2 - med)])
        [MAD1, MAD2] = S_s.reshape(-1)
     
        [Xmad1, Xmad2] = np.array([xnew1, xnew2])[
            np.argsort([np.abs(xnew1 - med), np.abs(xnew2 - med)])
        ].reshape(-1)
        [i_MAD1, i_MAD2] = np.array(index)[
            np.argsort([np.abs(xnew1 - med), np.abs(xnew2 - med)])
        ].reshape(-1)
    #print(np.array([MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2]).reshape(-1))
    par = np.round(np.array([MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2]).squeeze(), 10)

    return X, par, case



def move2odd(X, loc=None, scale=None, verbose=False, index=None, med=None, MAD=None, par=[],distribution="normal",shape=1):
    
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
            return np.round(med - MAD, 8),me
        
    def zone_odd_ab(xi, med, MAD):
        xi = np.round(xi, 8)

        if xi == med:
            return med, MAD
        elif np.round(xi, 8) == np.round(med + MAD, 8):
            return np.round(med + MAD, 8), np.round(med + MAD, 8)
        elif np.round(xi, 8) == np.round(med - MAD, 8):
            return np.round(med - MAD, 8), np.round(med - MAD, 8)
        elif xi < med - MAD:
            return -np.inf, np.round(med - MAD, 8)
        elif xi < med:
            return np.round(med - MAD, 8), med
        elif xi < med + MAD:
            return med, np.round(med + MAD, 8)
        else:
            return np.round(med + MAD_init, 8), np.inf

    
    
    n = len(X) // 2
    if index == None:
        index = np.random.choice(len(X), 2, replace=False)
    xij = np.round(X[index], 8)
    xij = X[index]
    xi, xj = xij[0], xij[1]
    a,b=0,0
    if len(par) == 0:
        if np.round (med +  MAD, 8) in X:
            xmad = np.round (med +  MAD, 8)
        elif np.round (med -  MAD, 8) in X:
            xmad = np.round (med -  MAD, 8)
        else:
            print("pas de mad ???")
        i_MAD = np.where(X == xmad)[0][0]
        par = [i_MAD, xmad]
    [i_MAD, xmad] = par
    med,xmad=np.round([med,xmad],8)

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
        xnew1, xnew2 = (
            truncated(
                a=(a - loc) / scale, b=(b - loc) / scale, size=1, loc=loc, scale=scale,distribution=distribution, shape=shape
            )[0],
            med,
        )
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
            xnew1, xnew2 = (
                truncated(
                    a=(a - loc) / scale, b=(b - loc) / scale, size=1, loc=loc, scale=scale,distribution=distribution, shape=shape)[0],
                xmad,
            )
        else:
            if np.abs(xother - med) > MAD:
                xnew1 = truncated_2inter(loc, scale, -np.inf, med -  MAD, med +  MAD, np.inf,distribution=distribution, shape=shape)[0]
                case = "3b"
            else:
                case = "3c"
                #case = "3c : {} devient {} dans R et {} devient {} dans [{},{}]".format(round(xi,3),round(xnew1,3),round(xj,3),round(xnew2,3),a,b)
                a, b = med -  MAD, med + MAD
                xnew1 = truncated(
                    a=(a - loc) / scale, b=(b - loc) / scale, size=1, loc=loc, scale=scale,distribution=distribution, shape=shape
                )[0]
            if xnew1 > med:
                xmad = np.round (med -  MAD,8)
            else:
                xmad = np.round (med +  MAD,8)
            xnew2 = xmad
            #print("xmad devient",xmad)
        i_MAD = index[1]
    else:
        if type(zone_odd(xi,med,MAD))!=str and type(zone_odd(xj,med,MAD))!=str: 
            if sorted([zone_odd(xi, med, MAD), zone_odd(xj, med, MAD)]) in [[1, 3], [2, 4]]:
                case = "4a"
                if distribution == "normal": xnew1 = np.random.normal(loc, scale, 1)[0]
                elif distribution == "cauchy": xnew1 = scipy.stats.cauchy(loc, scale).rvs(1)[0]
                elif distribution== "weibull": xnew1 = scipy.stats.weibull_min(c=shape, scale=scale, loc=loc).rvs(1)[0]
                elif distribution== "weibull2": xnew1 = scipy.stats.weibull_min(c=shape, scale=scale).rvs(1)[0]
                a, b = zone_odd_C_ab(xnew1, med, MAD)
                xnew2 = truncated(
                    a=(a - loc) / scale, b=(b - loc) / scale, size=1, loc=loc, scale=scale,distribution=distribution, shape=shape
                )[0]
                #case = "4a : {} devient {} dans R et {} devient {} dans [{},{}]".format(round(xi,3),round(xnew1,3),round(xj,3),round(xnew2,3),a,b)
            else:
                case = "4b"
                #case = "4b : {} devient {} dans [{},{}] et {} devient {} dans [{},{}]".format(round(xi,3),round(xnew1,3),a1,b1,round(xj,3),round(xnew2,3),a2,b2)
                a1, b1 = zone_odd_ab(xi, med, MAD)
                a2, b2 = zone_odd_ab(xj, med, MAD)
                xnew1, xnew2 = (
                    truncated(
                        a=(a1 - loc) / scale,
                        b=(b1 - loc) / scale,
                        size=1,
                        loc=loc,
                        scale=scale,
                        distribution=distribution,
                        shape=shape
                    )[0],
                    truncated(
                        a=(a2 - loc) / scale,
                        b=(b2 - loc) / scale,
                        size=1,
                        loc=loc,
                        scale=scale,
                        distribution=distribution,
                        shape=shape
                    )[0],
                )
        else : 
            print("ERREUR : xij = {} xmad = {} zone= {} med = {} MAD = {} m-s = {}, m+s = {}".format(xij,xmad,[zone_odd(xi,m,s),zone_odd(xj,m,s)],m,s,m-s,m+s))
            xnew1,xnew2=xi,xj
            case="erreur"
   # print(X[index],np.round(np.array([xnew1, xnew2]),8),case)
    X[index] = np.round(np.array([xnew1, xnew2]), 8).reshape(-1)

    return X, [i_MAD, np.round(xmad,8)], case


def medMAD(X):
    return (np.median(X), scipy.stats.median_abs_deviation(X))

def move2coords(
    X,
    mean=None,
    std=None,
    verbose=False,
    index=None,
    med=None,
    MAD=None,
    par=[],
    distribution="normal",
    shape=1
):
    if len(X) % 2 == 0:
        return move2even(
            X=X,
            loc=mean,
            scale=std,
            verbose=verbose,
            index=index,
            med=med,
            MAD=MAD,
            par=par,
            distribution=distribution,
            shape=shape
        )
    return move2odd(
        X=X,
        loc=mean,
        scale=std,
        verbose=verbose,
        index=index,
        med=med,
        MAD=MAD,
        par=par,
        distribution=distribution,
        shape=shape
    )


# def move_resample_zone(X, mean=None, std=None, med=None, MAD=None, par=[]):
#     if len(X) % 2 == 0:
#         return move_resample_zone_even(X=X, mean=mean, std=std, med=med, MAD=MAD, par=par)
#     return move_resample_zone_odd(X=X, mean=mean, std=std, med=med, MAD=MAD, par=par)


# def move_k(X, mean=None, std=None, med=None, MAD=None, par=[]):
#     if len(X) % 2 == 0:
#         return move_k_even(X=X, mean=mean, std=std, med=med, MAD=MAD, par=par)
#     return move_k_odd(X=X, mean=mean, std=std, med=med, MAD=MAD, par=par)


# def move_xMAD(X, mean=None, std=None, med=None, MAD=None, par=[]):
#     if len(X) % 2 == 0:
#         return move_Xmad_even(X=X, mean=mean, std=std, med=med, MAD=MAD, par=par)
#     return move_Xmad_odd(X=X, mean=mean, std=std, med=med, MAD=MAD, par=par)


def MAD_init(N, med, MAD, distribution):
    loc,scale,shape=0,1,1
    if distribution == "normal": Y = np.round(np.random.normal(loc, scale, N), 8)
    elif distribution == "cauchy": Y = np.round(scipy.stats.cauchy(loc=loc, scale=scale).rvs(N), 8)
    elif distribution == "weibull" or distribution == "weibull2": Y= np.round(scipy.stats.weibull_min(c=shape,loc=loc,scale=scale).rvs(N), 8)
    elif distribution== "lognormal": 
        if MAD > med: 
            print("MAD > med : impossible cas lognormal")
            return None
        n=N//2
        k=np.ceil(n/2)
        if N%2==0: return np.repeat([.99*med,1.01*med,med+MAD*.99,med+MAD*1.01,med-1.5*MAD,med-0.5*MAD,med+0.5*MAD,med+1.5*MAD],[1,1,1,1,n-k,k-1,n-k-2,k-1])
        return np.repeat([med,med+MAD,med-1.5*MAD,med-0.5*MAD,med+0.5*MAD,med+1.5*MAD],[1,1,n-k+1,k-1,n-k,k-1]),[]
    m_Y, s_Y = np.median(Y), scipy.stats.median_abs_deviation(Y)
    if distribution == "normal":
        init_par = [med,MAD*1.4826]
    elif distribution == "cauchy":
        init_par=[med,MAD]
    elif distribution=="weibull":
        shape1= 2
        scale1 = MAD/.48
        loc1 = med-scale1*np.log(2)
        init_par=[loc1,scale1,1]
    elif distribution=="weibull2":
        init_par=[0,scale / s_Y * MAD,shape]
        Y2=np.round((Y - m_Y) / s_Y * MAD+ med, 8)
        Y2=np.where(Y2>0,Y2,0.00001)
        return Y2,init_par
    return np.round((Y - m_Y) / s_Y * MAD+ med, 8),init_par

def Gibbs_med_MAD(T,N,med,MAD,distribution,par_prior=[0,1,1,1,1,1],std_prop1=0.1,std_prop2=0.1,std_prop3=0.1,List_X=False,verbose=True,perturb=True,init_X=[]):
    log_norm=False
    if init_X==[]:
        X_0,init_theta= MAD_init(N,med,MAD,distribution)
        #init_theta=[-10,2.6,10]
    else : 
        X_0=np.round(init_X,8)
        if distribution=="normal":init_theta=np.round([np.median(X_0),scipy.stats.median_abs_deviation(X_0)*1.4826],8)
        if distribution=="cauchy":init_theta=np.round([np.median(X_0),scipy.stats.median_abs_deviation(X_0)],8)
        
        if distribution=="weibull" or distribution=="weibull2":init_theta=np.round([np.median(X_0),scipy.stats.median_abs_deviation(X_0),1],8)
    Theta=[init_theta]
    X_list=[X_0]
    Mean=[np.mean(X_0)]
    Std=[np.std(X_0)]
    par = []
    Case=[]
    K=[np.sum(np.where(X_0>med+MAD,1,0))]
    if distribution=="weibull":
        loc,scale,shape=init_theta
    elif distribution=="weibull2":
        loc,scale,shape=init_theta
    else:
        loc,scale=init_theta
        shape=1
        
    
    #print(init_theta,medMAD(X_0))
    X=X_0.copy()
    
    for i in tqdm(range(T),disable=not(verbose)):
        
        if perturb: X,par,case=move2coords(X,loc,scale,med=med,MAD=MAD,distribution=distribution,par=par,shape=shape)
        
        if distribution=="normal":
            mu,tau=post_NG(X,par_prior)
            loc,scale=mu,1/np.sqrt(tau)
            theta=[loc,scale]
        elif distribution=="lognormal":
            mu,tau=post_NG(np.log(X),par_prior)
            loc,scale=mu,1/np.sqrt(tau)
            theta=[loc,scale]
        elif distribution=="cauchy":
            loc=post_cauchy_theta(Theta[-1][0],Theta[-1][1],X,par_prior[:2],std_prop1)
            scale=post_cauchy_gamma(loc,Theta[-1][1],X,par_prior[2:],std_prop2)
            theta=[loc,scale]
        
        elif distribution=="weibull":
            #print(np.min(X),loc)
            loc=post_weibull_loc(Theta[-1][0],Theta[-1][1],Theta[-1][2],X,par_prior[:2],std_prop1)
            scale=post_weibull_scale(loc,Theta[-1][1],Theta[-1][2],X,par_prior[2:4],std_prop2)
            shape=post_weibull_k(loc,scale,Theta[-1][2],X,par_prior[4:],std_prop3)
            theta=[loc,scale,shape]
            
        elif distribution=="weibull2":
            #print(np.min(X),loc)
            loc=0
            scale=post_weibull_scale(loc,Theta[-1][1],Theta[-1][2],X,par_prior[2:4],std_prop2)
            shape=post_weibull_k(loc,scale,Theta[-1][2],X,par_prior[4:],std_prop3)
            theta=[loc,scale,shape]
        
        Theta.append(theta)
        Mean.append(np.mean(X))
        Std.append(np.std(X))
        if perturb: Case.append(case)
        K.append(np.sum(np.where(X>med+MAD,1,0)))
    

        if List_X: X_list.append(X.copy())
        
    if not(List_X): X_list.append(X.copy())
    
    if log_norm:
        X_list=np.exp(np.array(X_list))
    
        
    if verbose and distribution=="cauchy":
        print("Acceptation rate of loc = {:.2%} and of scale = {:.2%}".format(len(np.unique(np.array(Theta)[:,0],axis=0))/len(Theta),len(np.unique(np.array(Theta)[:,1],axis=0))/len(Theta)))
    if verbose and distribution=="weibull" or distribution=="weibull2":
        print("Acceptation rate of loc = {:.2%}, of scale = {:.2%} and of shape = {:.2%}".format(len(np.unique(np.array(Theta)[:,0],axis=0))/len(Theta),len(np.unique(np.array(Theta)[:,1],axis=0))/len(Theta),len(np.unique(np.array(Theta)[:,2],axis=0))/len(Theta)))
    return {"X":X_list,"Mean":Mean,"Std":Std,"chains":np.array(Theta).T,"Case":Case,"K":K,"N":N,"med":med,"MAD":MAD,"distribution":distribution,"par_prior":par_prior,"T":T}
