# Insufficient Gibbs Sampling

The `InsufficientGibbs` package offers functionalities for sampling from posterior parameters when only robust and insufficient statistics of the data are available. A comprehensive explanation of the underlying theory can be found in the associated paper on arXiv (\url{link}{https://arxiv.org/abs/2307.14973}). Additionally, the code responsible for generating all figures presented in the paper is located in the `Figures` folder.
## `InsufficientGibbs` package


## Install

Install latest release via `pip`

```shell
pip install InsufficientGibbs
```

For latest development version clone the repository and install via pip

```shell
git clone https://github.com/antoineluciano/Insufficient-Gibbs-Sampling
cd src\InsufficientGibbs
pip install .
```

## Available distributions
* Normal (`Normal`)
* Cauchy (`Cauchy`)
* Laplace(`Laplace`)
* Gamma (`Gamma`) and its translated version (`TranslatedGamma`)
* LogNormal (`LogNormal`) and its translated version (`TranslatedLogNormal`)
* Weibull (`Weibull`) and its translated version (`TranslatedWeibull`)
* Generalized Pareto (`GeneralizedPareto`)

Add `Model` to the end of their function names to use them as models.

# Tutorial
## Create prior distributions


For each parameter of your model, create a variable of type `Distribution` representing its associated prior distribution. You must specify its hyperparameters and optionally its name.


## Create the model

Create a variable of type Model with your selected distribution and its predefined parameters as arguments.

## Sample from the posterior given insufficient statistics

To sample from the posterior of the model parameters given the observed statistics, you can use the three functions of the class `Model`: 
* `Gibbs_Quantile`: the observed data consists of a sequence of quantiles.
* `Gibbs_med_IQR`: the observed data consists of the median and the interquartile range (IQR).
* `Gibbs_med_MAD`: the observed data consists of the median and the median absolute deviation (MAD).

## Examples

### Cauchy distribution with Normal and Gamma priors

```python 
from InsufficientGibbs.Distribution import Normal, InverseGamma
from InsufficientGibbs.Model import CauchyModel

# Creating the prior distributions variables
mu = Normal(0,10, name= "x_0")
sigma = Gamma(2,2, name = "gamma")

# Creating the model variable
model = CauchyModel(mu,sigma)

T, N = 10000, 100

# Quantile case 
q1, med, q3 = -1, 0, 1
probs = [.25, .5, .75]
Cauchy_Q = model.Gibbs_Quantiles(T, N, [q1, med, q3], probs)

med, IQR = 0, 2
Cauchy_med_IQR = model.Gibbs_med_IQR(T, N, med, IQR) 

# Median, MAD case
med, MAD = 0, 1
Cauchy_med_MAD = model.Gibbs_med_MAD(T, N, med, MAD)
```
You can display the chain by using the `display_chains` function.


# How to add new distributions and models

As outlined in our paper, our method is applicable to all continuous distributions with compact support, as it only requires simulation according to a truncated version. Therefore, we provide guidance on how users can seamlessly integrate their own models by simply adding instances to the `Distribution` and `Model` classes.

For the `Distribution` class, an instance should include an initialization function `__init__`defining a variable `_distribution` containing all relevant distribution information (e.g., pdf, cdf, ppf) and a domain function describing the domain of definition.

Similarly, for the `Model` class, an instance should include an initialization function `__init__`, along with functions returning initialization parameters for the three studied scenarios (`Init_theta_Quantile`, `Init_theta_med_MAD`, and `Init_theta_med_IQR`).

To illustrate, we present examples of distributions. First, we showcase a distribution already implemented in SciPy (the Pareto distribution), followed by a custom implementation (the Pareto Type II distribution).


## Example of the Pareto Distribution

Add the the instance `Pareto` of the class `Distribution` in the file `Distribution.py`:

```python 
class Pareto(Distribution):
    """
    Container for Pareto Distribution

    Parameters
    ----------
    scale : float
        scale of the pareto
    shape : float
        shape of the pareto
    name : str
    """
    def __init__(self, 
            scale: float=1,
            shape: float=1,
            name: str="") -> None:
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = pareto(b=self.shape.value, scale=self.scale.value)
        parameters_dict = {'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)

    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["scale"], float('inf'))
```

Add the the instance `ParetoModel` of the class `Model` in the file `Models.py`:


```python    
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
```

## Example of the Pareto Type II distribution

First, define a class `pareto2` that plays the same role as the functions in SciPy.


```python 
class pareto2:
    def __init__(self, loc=0, scale=1,shape=1):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        
    def pdf(self,x):
        x=np.array(x)
        return np.where(x>=self.loc,(self.shape/self.scale)*(1+(x-self.loc)/self.scale)**(-self.shape-1),0)
    
    def cdf(self,x):
        x=np.array(x)
        return np.where(x>=self.loc,1-(1+(x-self.loc)/self.scale)**(-self.shape),0)
    def ppf(self,x):
        x=np.array(x)
        return self.loc+self.scale*((1-x)**(-1/self.shape)-1)
    def logpdf(self,x):
        x=np.array(x)
        return np.where(x>=self.loc,np.log(self.shape)-np.log(self.scale)-(self.shape+1)*np.log(1+(x-self.loc)/self.scale),-np.inf)
    def rvs(self,size):
        return self.ppf(np.random.random(size))
```

Then, add the instance `ParetoType2` of the class `Distribution` using the above class pareto2 in the file `Distribution.py`:

```python 
class ParetoType2(Distribution):
    
    def __init__(self,
            loc: float=0,
            scale: float=1,
            shape: float=1,
            name: str="",
            theta: list = []) -> None:
        if theta!=[]:
            loc,scale,shape = theta
        self.loc = ContinuousVariable(loc, name='loc')
        self.scale = PositiveContinuousVariable(scale, name='scale')
        self.shape = PositiveContinuousVariable(shape, name='shape')
        self.name = name
        self._distribution = pareto2(loc=self.loc.value, scale=self.scale.value,shape=self.shape.value)
        parameters_dict = {'loc': self.loc, 'scale':self.scale, 'shape':self.shape}
        super().__init__(parameters_dict, self.name)
        
        
    def domain(self) -> Tuple[float, float]:
        return (self.parameters_value["loc"], float('inf'))
```

Finally, add the instance `ParetoType2Model` of the class `Model` in the file `Models.py`:

```python 
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
```