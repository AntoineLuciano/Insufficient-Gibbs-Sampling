# Insufficient-Gibbs-Sampling


`InsufficientGibbs` is a package that enables users to sample from the posterior parameters when only certain robust statistics of the data are accessible. The paper that comprehensively describes the theory behind these methods can be found on arXiv (https://arxiv.org/abs/2307.14973). Additionally, the code responsible for generating all the paper figures can be located in the `figures` folder.

## Robust Gibbs package

We propose here three main functions named `Gibbs_med_MAD`, `Gibbs_med_IQR` and `Gibbs_Quantile` to cover the case when we observe the pairs (median, MAD) or (median, IQR) or a sequence of quantiles. 

## Install

Install latest release via `pip`

```shell
pip install InsufficientGibbs
```

For latest development version clone the repository and install via pip

```shell
git clone https://github.com/antoineluciano/Insufficient-Gibbs-Sampling
cd InsufficientGibbs
pip install .
```

## Available models/likelihood
* Normal distribution (`distribution="normal"`)
* Cauchy distribution (`distribution="cauchy"`)
* Weibull distribution (`distribution="weibull"`)
* Translated distribution (`distribution="translated_weibull"`)

## Available location priors
* Normal (`par_loc="normal"`)
* Cauchy (`par_loc="cauchy"`)
* Gamma (`par_loc="gamma"`)
## Available scale priors
* Gamma (`par_loc="gamma"`)
* Jeffreys (`par_loc="jeffreys"`)
## Available shape priors
* Gamma (`par_loc="gamma"`)


