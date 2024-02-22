# Insufficient Gibbs Sampling


`InsufficientGibbs` is a package that enables users to sample from the posterior parameters when only robust and insufficient statistics of the data are accessible. The paper that comprehensively describes the theory behind these methods can be found on arXiv (https://arxiv.org/abs/2307.14973). Additionally, the code responsible for generating all the paper figures can be located in the `Figures` folder.

## `InsufficientGibbs` package

We propose here three main functions named `Gibbs_med_MAD`, `Gibbs_med_IQR` and `Gibbs_Quantile` to cover the case when we observe the pairs (median, MAD) or (median, IQR) or a sequence of quantiles. 

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

Ajoute `Model` to the end of their function names to use them as models.

# Tutorial

## Create prior distributions

## Create the model

## Sample from the posterior of 

# How to add new distributions and models

