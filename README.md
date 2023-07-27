# Unsufficient-Gibbs-Sampling


`Robust_Gibbs` is a package that allows users to sample from the parameters posteriors when only some robust statistics of the data are available. The paper that describe all the theory of the methods can be found on arXiV (soon!). 
You can found the code that produces all the figures in the paper in the file `figures`. 

## Robust Gibbs package

We propose here three mains functions named `Gibbs_med_MAD`, `Gibbs_med_IQR` and `Gibbs_Quantile` to cover the case when we observe the pairs (median, MAD) or (median, IQR) or a sequence of quantiles. 


## Install

Install via clone the repository and install via pip

```shell
git clone https://github.com/???
pip install .
```

## Use

Here, we sample from the posterior of parameters of a normal distribution using the couple of conjuguate couple Normal-InverseGamma. 

```python
# CODE
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

## Available scale and shape priors
* Gamma (`par_loc="gamma"`)


