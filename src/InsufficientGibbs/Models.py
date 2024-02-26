# from InsufficientGibbs.Model import Model
# from InsufficientGibbs.Distribution import *

from Model import Model
from Distribution import *
from typing import Dict, Tuple
from scipy.stats import norm
import numpy as np

class NormalModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        if scale.name == "": scale.name = "scale"

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
        loc = Q[np.argmin(np.abs(P-.5))]
        scale = (Q[-1] - Q[0]) / (Normal(loc=loc)._distribution.ppf(P[-1]) - Normal(loc=loc)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale}
            
    def Init_theta_med_MAD(self, med, MAD):
        loc, scale = med, MAD/norm.ppf(.75)
        return {self.loc.name: loc, self.scale.name: scale}
        
    def Init_theta_med_IQR(self, med, IQR):
        loc, scale = med, IQR/(2*norm.ppf(.75))
        return {self.loc.name: loc, self.scale.name: scale}
    
class NormalKnownScaleModel(Model):
    def __init__(self, loc:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        self.loc = loc
        self.parameters_dict = {self.loc.name}
        super().__init__(self.parameters_dict)
        self.type_distribution = Normal_known_scale
        self.distrib_name = "normal_known_scale"
        self.init_method = "stable"
        
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = Q[np.argmin(np.abs(P-.5))]
        return {self.loc.name: loc}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc = med
        return {self.loc.name: loc}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc = med
        return {self.loc.name: loc}
        
class CauchyModel(Model):
    
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        if scale.name == "": scale.name = "scale"
        
        if loc.name == scale.name:
            raise ValueError("parameters must have different names.")
        
        self.loc = loc
        self.scale = scale
        self.type_distribution = Cauchy
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale}
        super().__init__(self.parameters_dict)
        self.distrib_name = "cauchy"
        self.init_method = "stable"

    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))

    def Init_theta_Quantile(self, Q, P):
        loc = Q[np.argmin(np.abs(P-.5))]
        scale = (Q[-1] - Q[0]) / (Cauchy(loc=loc)._distribution.ppf(P[-1]) - Cauchy(loc=loc)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc, scale = med, MAD
        return {self.loc.name: loc, self.scale.name: scale}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc, scale = med, IQR/2
        return {self.loc.name: loc, self.scale.name: scale}
    


class GammaModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        if scale.name == "": scale.name = "scale"
        if shape.name == "": shape.name = "shape"
        
        if scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.scale = scale
        self.shape = shape
        self.type_distribution = Gamma 
        self.parameters_dict = {self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)
        self.distrib_name = "gamma"
        self.init_method = "naive"

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (Gamma(shape=shape)._distribution.ppf(P[-1]) - Gamma(shape=shape)._distribution.ppf(P[0]))
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        scale = 1
        shape = 1
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        scale = 1
        shape = 1
        return {self.scale.name: scale, self.shape.name: shape}
    

class ReparametrizedGammaModel(Model):
    
    def __init__(self, mean:Distribution, std:Distribution) -> None:
        if mean.name == "": mean.name = "mean"
        if std.name == "": std.name = "std"
        
        if mean.name == std.name:
            raise ValueError("parameters must have different names.")
        self.mean = mean
        self.std = std
        self.type_distribution = ReparametrizedGamma 
        self.parameters_dict = {self.mean.name: self.mean, self.std.name: self.std}
        super().__init__(self.parameters_dict)  
        self.distrib_name = "reparametrized_gamma"
        self.init_method = "naive"
    
    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))

    def Init_theta_Quantile(self, Q, P):
        scale = (Q[len(Q) // 2])
        shape = (Q[-1] - Q[0]) / (Gamma(scale = scale, shape = 2)._distribution.ppf(P[-1]) - Gamma(scale = scale, shape = 2)._distribution.ppf(P[0]))
        mean = scale*shape
        std = np.sqrt(scale**2*shape)
        return {self.mean.name: mean, self.std.name: std}
    
    def Init_theta_med_MAD(self, med, MAD):
        mean = 1
        std = 1
        return {self.mean.name: mean, self.std.name: std}
    
    def Init_theta_med_IQR(self, med, IQR):
        mean = 1
        std = 1
        return {self.mean.name: mean, self.std.name: std}
    

class TranslatedGammaModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        if scale.name == "": scale.name = "scale"
        if shape.name == "": shape.name = "shape"
        
        if loc.name == scale.name or loc.name == shape.name or scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedGamma 
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)
        self.distrib_name = "translated_gamma"
        self.init_method = "naive"
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value[self.loc.name], float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = 2*Q[0]-Q[1]
        scale =  2
        scale = (Q[-1] - Q[0]) / (TranslatedGamma(loc=loc,shape=shape)._distribution.ppf(P[-1]) - TranslatedGamma(loc=loc,shape=shape)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc = med-2*MAD
        scale = 1
        shape = 1
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc = med-IQR
        scale = 1
        shape = 1
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}


class LogNormalModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        if scale.name == "": scale.name = "scale"
        if shape.name == "": shape.name = "shape"
        
        if scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.scale = scale
        self.shape = shape
        self.type_distribution = LogNormal 
        self.parameters_dict = {self.scale.name: self.scale, self.shape.name: self.shape}
        self.parameters_value = {self.scale.name: None, self.shape.name: None}
        super().__init__(self.parameters_dict)
        self.distrib_name = "lognormal"
        self.init_method = "naive"
        

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        scale = np.log(Q[np.argmin(np.abs(P-.5))])
        shape = (Q[-1] - Q[0]) / (LogNormal(shape=1, scale=np.exp(scale))._distribution.ppf(P[-1]) - LogNormal(shape=1, scale=np.exp(scale))._distribution.ppf(P[0]))
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        scale = np.log(med)
        shape = 1 
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        scale = np.log(med)
        shape = 1
        return {self.scale.name: scale, self.shape.name: shape}

class ReparametrizedLogNormalModel(Model):
        def __init__(self, mean:Distribution, std:Distribution) -> None:
            if mean.name == "": mean.name = "mean"
            if std.name == "": std.name = "std"
            
            if mean.name == std.name:
                raise ValueError("parameters must have different names.")
            self.mean = mean
            self.std = std
            self.type_distribution = ReparametrizedLogNormal 
            self.parameters_dict = {self.mean.name: self.mean, self.std.name: self.std}
            self.parameters_value = {self.mean.name: None, self.std.name: None}
            super().__init__(self.parameters_dict)
            self.distrib_name = "reparametrized_lognormal"
            self.init_method = "naive"
            
            
        def domain(self) -> Tuple[float, float]:
            return (0, float('inf'))
        
        def Init_theta_Quantile(self, Q, P):
            scale = np.log(Q[np.argmin(np.abs(P-.5))])
            shape = (Q[-1] - Q[0]) / (LogNormal(scale=scale, shape=1)._distribution.ppf(P[-1]) - LogNormal(scale=scale, shape=1)._distribution.ppf(P[0]))
            mean = np.exp(scale+shape**2/2)
            std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*scale+shape**2))
            return {self.mean.name: mean, self.std.name: std}
        
        def Init_theta_med_MAD(self, med, MAD):
            scale = np.log(med)
            shape = 1
            mean = np.exp(scale+shape**2/2)
            std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*scale+shape**2))
            return {self.mean.name: mean, self.std.name: std}
        
        def Init_theta_med_IQR(self, med, IQR):
            scale = np.log(med)
            shape = 1
            mean = np.exp(scale+shape**2/2)
            std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*scale+shape**2))
            return {self.mean.name: mean, self.std.name: std}


class TranslatedLogNormalModel(Model):
    
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        if scale.name == "": scale.name = "scale"
        
        if loc.name == scale.name or loc.name == shape.name or scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedLogNormal 
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale, self.shape.name: self.shape}
        self.parameters_value = {self.loc.name: None, self.scale.name: None, self.shape.name: None}
        super().__init__(self.parameters_dict)
        self.distrib_name = "translated_lognormal"
        self.init_method = "naive"
        
    
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value[self.loc.name], float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = 2*Q[0]-Q[1]
        scale = np.log(Q[np.argmin(np.abs(P-.5))]-loc)
        shape = (Q[-1] - Q[0]) / (TranslatedLogNormal(loc=loc, scale=scale, shape = 1)._distribution.ppf(P[-1]) - TranslatedLogNormal(loc=loc, scale = scale, shape = 1)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc = med-2*MAD
        scale = 1
        shape = 1
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}

    def Init_theta_med_IQR(self, med, IQR):
        loc = med-IQR
        scale = 1
        shape = 1
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    

class WeibullModel(Model):

    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        if scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.scale = scale
        self.shape = shape
        self.type_distribution = Weibull #to access corresponding distribution in fit
        self.parameters_dict = {self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)
        self.distrib_name = "weibull"
        self.init_method = "naive"

    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))

    def Init_theta_Quantile(self, Q, P):
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (Weibull(shape=shape)._distribution.ppf(P[-1]) - Weibull(shape=shape)._distribution.ppf(P[0]))
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        scale = 1
        shape = 1.5
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        scale = 1
        shape = 1.5
        return {self.scale.name: scale, self.shape.name: shape}
    
    
class TranslatedWeibullModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        if scale.name == "": scale.name = "scale"
        if shape.name == "": shape.name = "shape"
        
        if loc.name == scale.name or loc.name == shape.name or scale.name == shape.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = TranslatedWeibull 
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale, self.shape.name: self.shape}
        super().__init__(self.parameters_dict)
        self.distrib_name = "translated_weibull"
        self.init_method = "naive"
        

    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value[self.loc.name], float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = 2*Q[0]-Q[1]
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (TranslatedWeibull(loc=loc, shape=shape)._distribution.ppf(P[-1]) - TranslatedWeibull(loc=loc, shape=shape)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc = med-2*MAD
        scale = 1
        shape = 1.5
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc = med-IQR
        scale = 1
        shape = 1.5
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}

    
class GeneralizedParetoModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        if scale.name == "": scale.name = "scale"
        if shape.name == "": shape.name = "shape"
        
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
    
    def Init_theta_Quantile(self, Q, P):
        loc = 2*Q[0]-Q[1]
        shape = .1
        scale = (Q[-1] - Q[0]) / (GeneralizedPareto(loc=loc, scale=scale, shape=shape)._distribution.ppf(P[-1]) - GeneralizedPareto(loc=loc, scale=scale, shape=shape)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc = med-2*MAD
        scale = 1
        shape = .1
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc = med-IQR
        scale = 1
        shape = .1
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
   



class LaplaceModel(Model):
    def __init__(self, loc:Distribution, scale:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        if scale.name == "": scale.name = "scale"
        
        if loc.name == scale.name:
            raise ValueError("parameters must have different names.")
        self.loc = loc
        self.scale = scale
        self.type_distribution = Laplace
        self.parameters_dict = {self.loc.name: self.loc, self.scale.name: self.scale}
        super().__init__(self.parameters_dict)
        self.distrib_name = "laplace"
        self.init_method = "stable"
    
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = Q[np.argmin(np.abs(P-.5))]
        scale = (Q[-1] - Q[0]) / (Laplace(loc=loc)._distribution.ppf(P[-1]) - Laplace(loc=loc)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc, scale = med, MAD/np.log(2)
        return {self.loc.name: loc, self.scale.name: scale}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc, scale = med, IQR/(2*np.log(2))
        return {self.loc.name: loc, self.scale.name: scale}

class LaplaceKnownScaleModel(Model):
    def __init__(self, loc:Distribution) -> None:
        if loc.name == "": loc.name = "loc"
        
        self.loc = loc
        self.type_distribution = Laplace_known_scale
        self.parameters_dict = {self.loc.name: self.loc}
        super().__init__(self.parameters_dict)
        self.distrib_name = "laplace_known_scale"
        self.init_method = "stable"
        
    
    def domain(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = Q[np.argmin(np.abs(P-.5))]
        return {self.loc.name: loc}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc = med
        return {self.loc.name: loc}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc = med
        return {self.loc.name: loc}

    
class ParetoType2Model(Model):
    def __init__(self, loc:Distribution, scale:Distribution, shape:Distribution) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.type_distribution = ParetoType2
        self.parameters_dict = {loc.name: loc, scale.name: scale, shape.name: shape}
        super().__init__(self.parameters_dict)
        self.distrib_name = "pareto_type2"
        self.init_method = "naive"
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        loc = 2*Q[0]-Q[1]
        shape = 1.5
        scale = (Q[-1] - Q[0]) / (ParetoType2(loc=loc, scale=scale, shape=shape)._distribution.ppf(P[-1]) - ParetoType2(loc=loc, scale=scale, shape=shape)._distribution.ppf(P[0]))
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        loc = med-2*MAD
        scale = 1
        shape = 1.5
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        loc = med-IQR
        scale = 1
        shape = 1.5
        return {self.loc.name: loc, self.scale.name: scale, self.shape.name: shape}
    
    
class ParetoModel:
    def __init__(self, scale:Distribution, shape:Distribution) -> None:
        self.scale = scale
        self.shape = shape
        self.type_distribution = Pareto
        self.parameters_dict = {scale.name: scale, shape.name: shape}
        self.distrib_name = "pareto"
        self.init_method = "naive"
        
    def domain(self) -> Tuple[float, float]:
        return (0, float('inf'))
    
    def Init_theta_Quantile(self, Q, P):
        scale = 1
        shape = (Q[-1] - Q[0]) / (Pareto(scale=scale)._distribution.ppf(P[-1]) - Pareto(scale=scale)._distribution.ppf(P[0]))
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_MAD(self, med, MAD):
        scale = 1
        shape = 1.5
        return {self.scale.name: scale, self.shape.name: shape}
    
    def Init_theta_med_IQR(self, med, IQR):
        scale = 1
        shape = 1.5
        return {self.scale.name: scale, self.shape.name: shape}
    