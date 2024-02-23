from Model import Model
from Distribution import *
from typing import Dict, Tuple



class NormalModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        if loc.name == scale.name:
            raise ValueError("parameters must have different names.")
        
        self.loc = loc
        self.scale = scale
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale}
        super().__init__(self.parameters_dict)
        self.type_distribution = Normal
        self.distrib_name = "normal"
        self.init_method = "stable"
        
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (norm(loc=loc).ppf(P[-1]) - norm(loc=loc).ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale}
            
    def Init_theta_med_MAD(self, med, MAD):
        loc, scale = med, MAD/norm.ppf(.75)
        return {self.loc.name: loc, self.scale.name: scale}
        
    def Init_theta_med_IQR(self, med, IQR):
        loc, scale = med, IQR/(2*norm.ppf(.75))
        return {self.loc.name: loc, self.scale.name: scale}
    
class NormalKnownScaleModel(Model):
    def __init__(self, loc:Distribution) -> None:
        self.loc = loc
        self.parameters_dict = {self.loc.name}
        super().__init__(self.parameters_dict)
        self.type_distribution = Normal_known_scale
        self.distrib_name = "normal_known_scale"
        self.init_method = "stable"
        
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        self.parameters_value = {self.loc.name: loc}
        self._distribution = Normal_known_scale(loc=loc)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
    
    def med_IQR_Init(self, med, IQR, N, epsilon=0.001):
        loc = med
        self.parameters_value = {self.loc.name: loc}
        self._distribution = Normal_known_scale(loc=loc)
        X_0, Q_sim, Q_tot, I_order, G, I_sim = self.Init_X_med_IQR(N, med, IQR,epsilon=epsilon)
        return X_0, Q_sim, Q_tot, I_order, G, I_sim
        
class CauchyModel(Model):
    
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        if loc.name == scale.name:
            raise ValueError("parameters must have different names.")
        
        self.loc = loc
        self.scale = scale
        self.type_distribution = Cauchy
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        scale = MAD
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale}
        self._distribution = Normal(loc=loc, scale=scale)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (cauchy(loc).ppf(P[-1]) - cauchy(loc).ppf(P[0]))
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale}
        self._distribution = Cauchy(loc=loc, scale=scale)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G       
    
    def med_IQR_init(self, med, IQR, N, epsilon=0.001):
        loc = med
        scale = IQR/2
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale}
        self._distribution = Cauchy(loc=loc, scale=scale)
        X_0, Q_sim, Q_tot, I_order, G, I_sim = self.Init_X_med_IQR(N, med, IQR,epsilon=epsilon)
        return X_0, Q_sim, Q_tot, I_order, G, I_sim


class GammaModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        if scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.scale = scale
        self.shape = shape
        self.type_distribution = Gamma 
        self.parameters_dict = {self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (
            gamma(shape).ppf(P[-1])
            - gamma(shape).ppf(P[0])
        )
        self.parameters_value = {self.scale.name: scale, self.shape.name: shape}
        self._distribution = Gamma(scale=scale, shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G

class ReparametrizedGammaModel(Model):
    
    def __init__(self, mean:Distribution, std:Distribution) -> None:
        if mean.name == std.name:
            raise ValueError("parameters must have different names.")
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedGamma 
        self.parameters_dict = {self.mean.name: self.mean, self.std.name: self.std}
        super().__init__(self.parameters_dict)  
    
    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        scale = (Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (gamma(scale=scale, a=2).ppf(P[-1]) - gamma(scale=scale, a=2).ppf(P[0]))
        mean = scale*shape
        std = np.sqrt(scale**2*shape)
        self.parameters_value = {self.mean.name: mean, self.std.name: std}
        self._distribution = ReparametrizedGamma(mean=mean,std=std)
        return self.parameters_value
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

class TranslatedGammaModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == scale.name or loc.name == shape.name or scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedGamma 
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value[self.loc.name], float('inf'))

    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (gamma(loc=loc,scale=scale).ppf(P[-1]) - gamma(loc=loc,scale=scale).ppf(P[0]))
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
        self._distribution = TranslatedGamma(loc=loc,scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

class LogNormalModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        if scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.scale = scale
        self.shape = shape
        self.type_distribution = LogNormal 
        self.parameters_dict = {self.scale.name: self.scale, self.shape.name: self.shape}
        self.parameters_value = {self.scale.name: None, self.shape.name: None}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (lognorm(s=1, scale=np.exp(scale)).ppf(P[-1]) - lognorm(s=1, scale=np.exp(scale)).ppf(P[0]))
        
        self.parameters_value = {self.scale.name: scale, self.shape.name: shape}
        self._distribution = LogNormal(scale=scale, shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
class ReparametrizedLogNormalModel(Model):
    
        def __init__(self, mean:Distribution, std:Distribution) -> None:
            if mean.name == std.name:
                raise ValueError("parameters must have different names.")
            self.mean = mean
            self.std = std
            self.type_distribution = ReparametrizedLogNormal 
            self.parameters_dict = {self.mean.name: self.mean, self.std.name: self.std}
            self.parameters_value = {self.mean.name: None, self.std.name: None}
            super().__init__(self.parameters_dict)
            
        def domain(self) -> Tuple[float, float]:
            return (0, float('inf'))
        
        def Quantile_Init(self, Q, P, N, epsilon=0.001):
            scale = np.log(Q[len(Q) // 2])
            shape = (Q[-1] - Q[0]) / (lognorm(s=1, scale=np.exp(scale)).ppf(P[-1]) - lognorm(s=1, scale=np.exp(scale)).ppf(P[0]))
            mean = np.exp(scale+shape**2/2)
            std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*scale+shape**2))
            self.parameters_value = {self.mean.name: mean, self.std.name: std}
            self._distribution = ReparametrizedLogNormal(mean=mean,std=std)
            X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
            return X_0, Q, Q_tot, Q_sim, I, I_sim, G

class TranslatedLogNormalModel(Model):
    
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == scale.name or loc.name == shape.name or scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedLogNormal 
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale, self.shape.name: self.shape}
        self.parameters_value = {self.loc.name: None, self.scale.name: None, self.shape.name: None}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value[self.loc.name], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        scale = np.log(Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[-1]) - lognorm(loc=loc,scale=np.exp(scale), s=1).ppf(P[0]))
    
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
        print(self.parameters_value)
        self._distribution = TranslatedLogNormal(loc=loc,scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    

    

class WeibullModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        if scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.scale = scale
        self.shape = shape
        self.type_distribution = Weibull #to access corresponding distribution in fit
        self.parameters_dict = {self.scale.name: self.scale, self.shape.name: self.shape}
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
        self.parameters_value = {self.scale.name: scale, self.shape.name: shape}
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G 
    
class TranslatedWeibullModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == scale.name or loc.name == shape.name or scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedWeibull 
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)

    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value[self.loc.name], float('inf'))
    
    def Quantile_Init(self, Q, P, N,init_theta=[], epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (
            weibull_min(shape, loc=loc).ppf(P[-1])
            - weibull_min(shape, loc=loc).ppf(P[0])
        )
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
        self._distribution = TranslatedWeibull(loc=loc, scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G 
    
class GeneralizedParetoModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == scale.name or loc.name == shape.name or scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = GeneralizedPareto
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value[self.loc.name], float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = 2*Q[0]-Q[1]
        shape = .1
        scale = (Q[-1] - Q[0]) / (self.type_distribution(loc=loc, shape=shape)._distribution.ppf(P[-1]) -  self.type_distribution(loc=loc, shape=shape)._distribution.ppf(P[0]))
        
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
        self._distribution = GeneralizedPareto(loc=loc, scale=scale,shape=shape)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon)
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G


class LaplaceModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        if loc.name == scale.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.type_distribution = Laplace
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Quantile_Init(self, Q, P, N, epsilon=0.001):
        loc = Q[len(Q) // 2]
        scale = (Q[-1] - Q[0]) / (laplace(loc).ppf(P[-1]) - laplace(loc).ppf(P[0]))
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale}
        self._distribution = Laplace(loc=loc, scale=scale)
        X_0, Q, Q_tot, Q_sim, I, I_sim, G = self.Init_X_Quantile(Q, P, N, epsilon=epsilon) 
        return X_0, Q, Q_tot, Q_sim, I, I_sim, G
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        scale = MAD/np.log(2)
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale}
        self._distribution = Laplace(loc=loc, scale=scale)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
    
    def med_IQR_Init(self, med, IQR, N, epsilon=0.001):
        loc = med
        scale = IQR/(2*np.log(2))
        self.parameters_value = {self.loc.name: loc, self.scale.name: scale}
        self._distribution = Laplace(loc=loc, scale=scale)
        X_0, Q_sim, Q_tot, I_order, G, I_sim = self.Init_X_med_IQR(N, med, IQR,epsilon=epsilon)
        return X_0, Q_sim, Q_tot, I_order, G, I_sim

class LaplaceKnownScaleModel(Model):
    def __init__(self, loc:Distribution) -> None:
        self.loc = loc
        self.type_distribution = Laplace_known_scale
        self.parameters_dict = {self.loc.name: self.loc}
        super().__init__(self.parameters_dict)
    
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def med_MAD_Init(self, med, MAD, N):
        loc = med
        self.parameters_value = {self.loc.name: loc}
        self._distribution = Laplace_known_scale(loc=loc)
        X_0 = self.Init_X_med_MAD_loc_scale_stable(N, med, MAD)
        return X_0
    
    def med_IQR_Init(self, med, IQR, N, epsilon=0.001):
        loc = med
        self.parameters_value = {self.loc.name: loc}
        self._distribution = Laplace_known_scale(loc=loc)
        X_0, Q_sim, Q_tot, I_order, G, I_sim = self.Init_X_med_IQR(N, med, IQR,epsilon=epsilon)
        return X_0, Q_sim, Q_tot, I_order, G, I_sim
    
    
