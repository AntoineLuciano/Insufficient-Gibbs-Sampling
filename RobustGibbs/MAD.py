import numpy as np
from scipy.stats import median_abs_deviation,norm,cauchy,weibull_min
from tqdm import tqdm


from truncated import *
from postertior_sample import *

def medMAD(X):
    return (np.median(X), median_abs_deviation(X))

def MAD_init(N, med, MAD, distribution):
    loc, scale, shape = 0, 1, 1.5
        
        
    if distribution in ["lognormal","weibull"]:
        if MAD > med:
            raise Exception("ERROR: MAD > med impossible for {} distribution !".format(distribution))
        n = N // 2
        k = np.ceil(n / 2)
        if distribution == "lognormal":
            init_theta = [np.log(med), MAD / med,None]
            par_names=["loc","scale"]
        elif distribution=="weibull":
            init_theta = [0,med/np.log(2),shape]
            par_names=["scale","shape"]
        
        if N % 2 == 0:
            return np.repeat(
                [
                    med-.01*MAD,
                    med+.01*MAD,
                    med + MAD * 0.99,
                    med + MAD * 1.01,
                    med - 1.5 * MAD,
                    med - 0.5 * MAD,
                    med + 0.5 * MAD,
                    med + 1.5 * MAD,
                ],
                [1, 1, 1, 1, n - k, k - 1, n - k - 2, k - 1],
            ),init_theta,par_names
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
            ),init_theta,par_names
    if distribution == "normal":
        Z = norm(loc=loc, scale=scale).rvs(N)
        par_names=["loc","scale"]
    elif distribution == "cauchy":
        Z =cauchy(loc=loc, scale=scale).rvs(N)
        par_names=["loc","scale"]
    elif distribution == "translated_weibull":
        Z =weibull_min(c=shape, loc=loc, scale=scale).rvs(N)
        par_names=["loc","scale","shape"]
    else : raise Exception("ERROR: distribution {} not implemented !".format(distribution))
    m_Z, s_Z = medMAD(Z)
    X_0=np.round((Z - m_Z) / s_Z * MAD + med, 8)
    
    if distribution == "normal":
        init_theta = [med, MAD * 1.4826,shape]
    elif distribution == "cauchy":
        init_theta = [med, MAD,shape]
    elif distribution == "translated_weibull":
        init_theta = [(loc- m_Z) / s_Z * MAD + med, scale*MAD/s_Z, shape]

    return X_0, np.round(init_theta,8),par_names


def resampling_even(
    X,
    theta,
    index=None,
    med=None,
    MAD=None,
    par=[],
    distribution="normal",
):
    def sym(m, x):
        return 2 * m - x

    def zone_even(xi, X, med=None, MAD=None, par=[]):
        X = np.array(X)
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
        par = np.round(par, 8)
        [MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2] = par
        
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
        elif np.round(med - MAD1,8) < xi < med1:
            return 2
        elif med2 < xi < np.round(med + MAD1,8):
            return 3
        elif xi > np.round(med + MAD2,8):
            return 4
        else:
            print(
                "PAS DE ZONE pour {}, m-MAD2 = {}, m-MAD1 ={}, m1 = {}, m2 = {}, m+MAD1 = {}, m+MAD2 = {}".format(
                    xi, med - MAD2, med - MAD1, med1, med2, med + MAD1, med + MAD2
                )
            )

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
        
    loc,scale,shape=theta
    if index == None:
        index = np.random.choice(len(X), 2, replace=False)
    X = np.array(X)
    xij = np.round(X[index],8)
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

    par = np.round(par, 8)

    MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2 = np.round(par,8)
    change_med = False
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
            distribution=distribution,
            shape=shape,
        )[0]

        xnew2 = sym(med, xnew1)

        change_med = True
    elif sorted(xij) == sorted([Xmad1, Xmad2]):
        S = np.sort(np.abs(X - med))
        epsilon = np.minimum(MAD1 - S[n - 2], S[n + 1] - MAD2)
        if xi < med and xj < med:
            case = "2b"
            a, b = med - MAD2 - epsilon, med - MAD1 + epsilon
            if a >= b: raise Exception("ERROR in med,MAD perturbation (case 2b) !")
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution,
                shape=shape,
            )[0]
            xnew2 = sym(med - MAD, xnew1)

        elif xi > med and xj > med:
            case = "2a"
            a, b = med + MAD1 - epsilon, med + MAD2 + epsilon
            if a >= b:
                raise Exception("ERROR in med,MAD perturbation (case 2a) !")
   
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution,
                shape=shape,
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
            if a1 >= b1 or a2 >= b2:
                raise Exception("ERROR in med,MAD perturbation (case 2c) !")
    
            xnew1 = truncated_2inter(
                loc, scale, a1, b1, a2, b2, distribution=distribution, shape=shape
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
        if xi in [med1, med2]: xm, xother = xi, xj
        elif xj in [med1, med2]: xm, xother = xj, xi
        else: raise Exception("ERROR in med,MAD perturbation (case 4) !")
        if xm == med1 and med + MAD1 > xother > med:
            case = "4a"
            a, b = med, med + MAD1
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution,
                shape=shape,
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
                distribution=distribution,
                shape=shape,
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
                distribution=distribution,
                shape=shape,
            )[0]
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
            xnew1 = truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution,
                shape=shape,
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
                distribution=distribution,
                shape=shape,
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
            xnew1 = truncated_2inter(loc, scale, a1, b1, a2, b2, shape=shape)[0]
            if a2 <= xnew1 <= b2:
                xnew2 = sym(med, xmad)
                if xmad == Xmad1:
                    Xmad1 = sym(med, Xmad1)
                elif xmad == Xmad2:
                    Xmad2 = sym(med, Xmad2)
                else: raise Exception("ERROR in med,MAD perturbation (case 5c) !")
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
                distribution=distribution,
                shape=shape,
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
                distribution=distribution,
                shape=shape,
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
                    distribution=distribution,
                    shape=shape,
                )[0]
            elif xnew1 > med + MAD2:
                xnew2 = truncated(
                    a=(med2 - loc) / scale,
                    b=(med + MAD1 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution,
                    shape=shape,
                )[0]
            elif med1 > xnew1 > med - MAD1:
                xnew2 = truncated(
                    a=-np.inf,
                    b=(med - MAD2 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution,
                    shape=shape,
                )[0]
            else:
                xnew2 = truncated(
                    a=(med + MAD2 - loc) / scale,
                    b=np.inf,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution,
                    shape=shape,
                )[0]
        elif sort_zone == [2, 3]:
            case = "6b "
            xnew1 = truncated(
                a=(med - MAD1 - loc) / scale,
                b=(med + MAD1 - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution,
                shape=shape,
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
                    distribution=distribution,
                    shape=shape,
                )[0]
            else:
                xnew2 = truncated(
                    a=(med - MAD1 - loc) / scale,
                    b=(med1 - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution,
                    shape=shape,
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
                    distribution=distribution,
                    shape=shape,
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
                distribution=distribution,
                shape=shape,
            )[0]
            a2, b2 = zone_even_ab(xj, med1, med2, MAD1, MAD2)
            xnew2 = truncated(
                a=(a2 - loc) / scale,
                b=(b2 - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution,
                shape=shape,
            )[0]

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
            np.argsort([np.abs(xnew1 - med), np.abs(xnew2 - med)])]
        
    par = np.round(
        np.array([MAD1, MAD2, Xmad1, Xmad2, i_MAD1, i_MAD2, med1, med2]).squeeze(), 8
    )

    return X, par, case

## ODD CASE 

def resampling_odd(
    X,
    theta=None,
    index=None,
    med=None,
    MAD=None,
    par=[],
    distribution="normal",
):
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
    
    loc,scale,shape=theta
    if index == None:
        index = np.random.choice(len(X), 2, replace=False)
    xij = np.round(X[index], 8)
    xij = X[index]
    xi, xj = xij[0], xij[1]
    a, b = 0, 0
    if len(par) == 0:
        if np.round(med + MAD, 8) in X:
            xmad = np.round(med + MAD, 8)
        elif np.round(med - MAD, 8) in X:
            xmad = np.round(med - MAD, 8)
        else:
            raise Exception("No MAD found!")
        i_MAD = np.where(X == xmad)[0][0]
        par = [i_MAD, xmad]
    [i_MAD, xmad] = par
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
        xnew1, xnew2 = (
            truncated(
                a=(a - loc) / scale,
                b=(b - loc) / scale,
                size=1,
                loc=loc,
                scale=scale,
                distribution=distribution,
                shape=shape,
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
                    a=(a - loc) / scale,
                    b=(b - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution,
                    shape=shape,
                )[0],
                xmad,
            )
        else:
            if np.abs(xother - med) > MAD:
                xnew1 = truncated_2inter(
                    loc,
                    scale,
                    -np.inf,
                    med - MAD,
                    med + MAD,
                    np.inf,
                    distribution=distribution,
                    shape=shape,
                )[0]
                case = "3b"
            else:
                case = "3c"
                # case = "3c : {} devient {} dans R et {} devient {} dans [{},{}]".format(round(xi,3),round(xnew1,3),round(xj,3),round(xnew2,3),a,b)
                a, b = med - MAD, med + MAD
                xnew1 = truncated(
                    a=(a - loc) / scale,
                    b=(b - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution,
                    shape=shape,
                )[0]
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
                if distribution == "normal":
                    xnew1 = norm(loc=loc, scale=scale).rvs(1)[0]
                elif distribution == "cauchy":
                    xnew1 = cauchy(loc=loc, scale=scale).rvs(1)[0]
                elif distribution == "weibull":
                    xnew1 = weibull_min(c=shape, scale=scale, loc=loc).rvs(1)[0]
                elif distribution == "translated_weibull":
                    xnew1 = weibull_min(c=shape, scale=scale).rvs(1)[0]
                a, b = zone_odd_C_ab(xnew1, med, MAD)
                xnew2 = truncated(
                    a=(a - loc) / scale,
                    b=(b - loc) / scale,
                    size=1,
                    loc=loc,
                    scale=scale,
                    distribution=distribution,
                    shape=shape,
                )[0]
            else:
                case = "4b"
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
                        shape=shape,
                    )[0],
                    truncated(
                        a=(a2 - loc) / scale,
                        b=(b2 - loc) / scale,
                        size=1,
                        loc=loc,
                        scale=scale,
                        distribution=distribution,
                        shape=shape,
                    )[0],
                )
        else:
            raise Exception("ERROR in med,MAD perturbation !")
            xnew1, xnew2 = xi, xj
            case = "erreur"
    # print(X[index],np.round(np.array([xnew1, xnew2]),8),case)
    X[index] = np.round(np.array([xnew1, xnew2]), 8).reshape(-1)

    return X, [i_MAD, np.round(xmad, 8)], case


def resampling_med_MAD(
    X,
    theta=None,
    index=None,
    med=None,
    MAD=None,
    par=[],
    distribution="normal",
):
    if len(X) % 2 == 0:
        return resampling_even(
            X=X,
            theta=theta,
            index=index,
            med=med,
            MAD=MAD,
            par=par,
            distribution=distribution,
        )
    return resampling_odd(
        X=X,
        theta=theta,
        index=index,
        med=med,
        MAD=MAD,
        par=par,
        distribution=distribution,
        )

### GIBBS SAMPLER

def Gibbs_med_MAD(
    T : int,
    N: int,
    med : float,
    MAD : float,
    distribution: str="normal",
    prior_loc:str="normal",
    prior_scale:str="gamma",
    prior_shape:str="gamma",
    par_prior_loc:list=[0, 1],
    par_prior_scale:list=[0, 1],
    par_prior_shape:list=[0, 1],
    std_prop_loc:float=0.1,
    std_prop_scale:float=0.1,
    std_prop_shape:float=0.1,
    List_X:bool=False,
    verbose:bool=True,
)-> dict:
    """Gibbs sampler to sample from the posterior of model parameters given the median and MAD of the data.

   Args:
    T (int): Number of iterations.
    N (int): Size of the vector X. 
    med (float): Observed median.
    MAD (float): Observed MAD (Median Absolute Deviation)
    distribution (str): Distribution of the data ("normal", "cauchy", "weibull", or "translated_weibull").
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
        ... input parameters
"""

    
    med,MAD=np.round([med,MAD],8)
    X_0,init_theta,par_names=MAD_init(N, med, MAD, distribution)
    Theta = [init_theta]
    X_list = [X_0]
    par = []
    # K=[np.sum(np.where(X_0>med+MAD,1,0))]

    X = X_0.copy()

    for i in tqdm(range(T), disable=not (verbose)):
        X, par, case = resampling_med_MAD(
            X,
            Theta[-1],
            med=med,
            MAD=MAD,
            distribution=distribution,
            par=par,
            )

        theta = posterior(
            X,
            Theta[-1],
            distribution, 
            prior_loc,
            prior_scale,
            prior_shape,
            par_prior_loc,
            par_prior_scale,
            par_prior_shape,
            std_prop_loc,
            std_prop_scale,
            std_prop_shape,
        )
        Theta.append(theta)
        if List_X:X_list.append(X.copy())
        
    if not (List_X):X_list.append(X.copy())
    
    Theta = np.array(Theta).T
    chains0={par_name:Theta[i] for i,par_name in enumerate(["loc","scale","shape"])}
    chains = {par_name: chains0[par_name] for par_name in par_names}
    if verbose and prior_loc!="NIG":
        
        acceptation_rate=[(len(np.unique(chains[par_name]))-1)/T for par_name in par_names]
        print('Acceptation rates MH :',end=" ")
        for i in range(len(par_names)):
            print("{} = {:.2%}".format(par_names[i],acceptation_rate[i]),end=" ")
        print()
    
    return {
        "X": X_list,
        "chains": chains,
        "N": N,
        "med": med,
        "MAD": MAD,
        "distribution": distribution,
        "prior_loc":prior_loc,
        "prior_scale": prior_scale,
        "prior_shape": prior_shape,
        "par_prior_loc": par_prior_loc,
        "par_prior_scale": par_prior_scale,
        "par_prior_shape": par_prior_shape,
        
        "T": T,
    }
