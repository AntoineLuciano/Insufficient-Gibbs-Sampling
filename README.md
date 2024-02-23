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


# How to add new distributions and models

To make it possible to use this new method with models not implemented in the package, users can easily add their own models by simply adding an instance to the Distribution and Model classes.

## Example of the Pareto Type II distribution

