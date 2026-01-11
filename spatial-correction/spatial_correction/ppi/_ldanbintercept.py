import os
from typing import Union

import flax.linen as nn
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.distributions import Categorical, NegativeBinomial2, Normal
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ._template import PPIAbstractClass
from ._utils import _zstat_generic2, optimize_ppi, optimize_ppi_gd


class NBModelIntercept(nn.Module):
    n_classes: int
    n_features: int
    mu_prior_std: Union[float, jnp.ndarray]
    n_obs_real: int
    mu_prior0_std: float

    def setup(self):
        self.pi_raw = self.param("pi", nn.initializers.normal(), (self.n_classes - 1,))
        self.mu0 = self.param("mu0", nn.initializers.normal(), (self.n_features))
        self.mu = self.param(
            "mu", nn.initializers.normal(), (self.n_classes - 1, self.n_features)
        )
        self.rho = self.param("rho", nn.initializers.normal(), (self.n_features,))

    def __call__(self, x, y):
        y_ = y.astype(jnp.int32)
        pi = jnp.concatenate([self.pi_raw, -jnp.sum(self.pi_raw, keepdims=True)])
        mu_placeholder = jnp.zeros_like(self.mu0)
        mu = jnp.concatenate([mu_placeholder[None], self.mu], axis=0)
        y_oh = jnp.eye(self.n_classes)[y_]
        mus_ = y_oh @ mu + self.mu0
        rates = jnp.exp(mus_)
        concentrations = jnp.exp(self.rho)

        # Categorical
        log_pc = Categorical(logits=pi).log_prob(y_)
        log_px_c = (
            NegativeBinomial2(mean=rates, concentration=concentrations)
            .log_prob(x)
            .sum(axis=-1)
        )

        # Regularization & priors
        # reg1 = -jnp.sum(self.pi_raw**2)
        if self.mu_prior0_std is not None:
            reg0 = (
                1.0
                / self.n_obs_real
                * Normal(loc=0.0, scale=self.mu_prior0_std).log_prob(self.mu0).sum()
            )
        else:
            reg0 = 0.0

        reg1 = 0.0
        if self.mu_prior_std is not None:
            reg2 = (
                1.0
                / self.n_obs_real
                * Normal(loc=0.0, scale=self.mu_prior_std).log_prob(self.mu).sum()
            )
        else:
            reg2 = 0.0
        priors = reg0 + reg1 + reg2

        loss = -(log_pc + log_px_c + priors)
        return {
            "loss": loss,
            "log_pc": log_pc,
            "log_px_c": log_px_c,
        }


class NBIntercept(PPIAbstractClass):
    def __init__(
        self,
        mu_prior_std=None,
        weight_init="zero",
        optimizer="gd",
        optimizer_kwargs=None,
        mu_prior0_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl
        all_y = np.hstack([y_gt, y_hat, y_unl])
        unique_y = np.unique(all_y)
        self.n_classes = unique_y.shape[0]
        necessary_range = np.arange(self.n_classes)
        assert np.isin(necessary_range, unique_y).all()

        self.inputs_gt = (x_gt, y_gt)
        self.inputs_hat = (x_hat, y_hat)
        self.inputs_unl = (x_unl, y_unl)
        n_obs_real = x_gt.shape[0]

        self.n_features = x_gt.shape[1]

        nparams_mu = (self.n_classes - 1) * self.n_features
        nparams_mu0 = self.n_features
        nparams_pi = self.n_classes - 1
        nparams_rho = self.n_features
        self.n_params = nparams_mu + nparams_mu0 + nparams_pi + nparams_rho
        self.n_params_mu = nparams_mu
        self.n_params_mu0 = nparams_mu0
        self.n_params_pi = nparams_pi
        self.n_params_rho = nparams_rho

        self.model = NBModelIntercept(
            n_classes=self.n_classes,
            n_features=self.n_features,
            mu_prior_std=mu_prior_std,
            n_obs_real=n_obs_real,
            mu_prior0_std=mu_prior0_std,
        )
        self.model_params = None
        if weight_init == "zero":
            self.zero_init()
        else:
            raise ValueError("invalid `weight_init`")
        self.lambd_ = None
        self.log = None
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

    def fit(self, lambd_=None):
        if lambd_ is None:
            lambd_ = self.get_lambda()
            self.lambd_ = lambd_
        self.get_asymptotic_distribution(lambd_)

    def get_pointestimate(self, lambd_):
        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl

        model_params0 = self.model_params if self.model_params is not None else None
        if self.optimizer == "lbfgs":
            model_params = optimize_ppi(
                self.model,
                lambd_=lambd_,
                x_gt=x_gt,
                y_gt=y_gt,
                x_hat=x_hat,
                y_hat=y_hat,
                x_unl=x_unl,
                y_unl=y_unl,
                model_params0=model_params0,
                **self.optimizer_kwargs
            )
        elif self.optimizer == "gd":
            model_params = optimize_ppi_gd(
                self.model,
                lambd_=lambd_,
                x_gt=x_gt,
                y_gt=y_gt,
                x_hat=x_hat,
                y_hat=y_hat,
                x_unl=x_unl,
                y_unl=y_unl,
                model_params0=model_params0,
                **self.optimizer_kwargs
            )
        self.model_params = model_params

        mu = np.array(model_params["params"]["mu"].reshape(-1))
        mu0 = np.array(model_params["params"]["mu0"])
        pi = np.array(model_params["params"]["pi"])
        rho = np.array(model_params["params"]["rho"])
        return np.hstack([mu, mu0, pi, rho])

    def grad_fn(self, inputs, batch_size=128):
        x, y = inputs
        n_obs = x.shape[0]

        @jax.jit
        def likelihood(model_params, x, y):
            return self.model.apply(model_params, x, y)["loss"]

        all_grads = np.zeros((n_obs, self.n_params))
        for i in tqdm(range(0, n_obs, batch_size), desc="Gradient computation"):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            n_obs_batch = x_batch.shape[0]
            score = jax.jit(jax.jacfwd(likelihood))
            grads = score(self.model_params, x_batch, y_batch)
            grad_mu = np.array(grads["params"]["mu"].reshape(n_obs_batch, -1))
            grad_mu0 = np.array(grads["params"]["mu0"].reshape(n_obs_batch, -1))
            grad_pi = np.array(grads["params"]["pi"].reshape(n_obs_batch, -1))
            grad_rho = np.array(grads["params"]["rho"].reshape(n_obs_batch, -1))
            all_grads[i : i + batch_size] = np.hstack(
                [grad_mu, grad_mu0, grad_pi, grad_rho]
            )
        return np.array(all_grads)

    def _construct_contrast(self, feature_id, idx_a):
        mu_contrast = np.zeros((self.n_classes - 1, self.n_features))
        mu_contrast[idx_a - 1, feature_id] = 1.0

        mu0_contrast = np.zeros(self.n_features)
        pi_contrast = np.zeros(self.n_classes - 1)
        rho_contrast = np.zeros(self.n_features)
        contrast = np.hstack(
            [mu_contrast.flatten(), mu0_contrast, pi_contrast, rho_contrast]
        )
        return contrast

    def construct_contrast(self, idx_a):
        _contrast = [
            self._construct_contrast(feature_id, idx_a)
            for feature_id in range(self.n_features)
        ]
        _contrast = np.vstack(_contrast)
        return _contrast

    def get_beta(self, idx_a):
        if idx_a == 0:
            raise ValueError("`class_a` cannot be the reference class.")

        contrast = self.construct_contrast(idx_a)
        beta = contrast @ self.theta
        cov = contrast @ self.sigma @ contrast.T
        return beta, cov, contrast

    def differential_expression(
        self, idx_a, feature_names=None, filter_pval_thresh=3e-2
    ):
        """
        Compute differential expression for a given class.

        Parameters
        ----------
        idx_a: int
            Index of the class to compute differential expression for.
        feature_names: list of str, optional
            Feature names.
        filter_pval_thresh: float, optional
            Set pvalues to 1.0 for features for which the Hessian is below this threshold.
        """
        beta, cov, contrast = self.get_beta(idx_a)
        _, pval = _zstat_generic2(beta, np.sqrt(np.diag(cov)), alternative="two-sided")

        if filter_pval_thresh is not None:
            is_hess_below_thresh = (
                np.diag(contrast @ self.hessian @ contrast.T) < filter_pval_thresh
            )
            pval[is_hess_below_thresh] = 1.0
        pval[np.isnan(pval)] = 1.0
        padj = multipletests(pval, method="fdr_bh")[1]
        is_de = padj < 0.05
        res = pd.DataFrame(
            {
                "beta": beta,
                "pval": pval,
                "padj": padj,
                "is_significant_005": is_de,
            }
        )
        if feature_names is not None:
            res["feature_name"] = feature_names
        return res

    def zero_init(self):
        mu = np.zeros((self.n_classes - 1, self.n_features))
        mu0 = np.zeros(self.n_features)
        pi0 = np.zeros(self.n_classes - 1)
        theta0 = np.zeros(self.n_features)
        params = {
            "params": {
                "mu": jnp.array(mu, dtype=jnp.float64),
                "mu0": jnp.array(mu0, dtype=jnp.float64),
                "pi": jnp.array(pi0, dtype=jnp.float64),
                "rho": jnp.array(theta0, dtype=jnp.float64),
            }
        }
        self.model_params = params

    def hessian_fn(self, inputs, device=None):
        x, y = inputs

        if device is None:
            device = jax.devices("cpu")[0]
        model_params_ = jax.device_put(self.model_params, device)
        # x_ = jax.device_put(x, device)
        # y_ = jax.device_put(y, device)
        n_obs = x.shape[0]
        obs_ids = np.arange(n_obs)
        model_ = self.model

        # @jax.jit
        def likelihood(model_params, x, y):
            return model_.apply(model_params, x, y)["loss"]

        hess_fn = jax.hessian(likelihood)

        def get_hess_block(hess, param1, param2, shape1, shape2):
            return (
                hess["params"][param1]["params"][param2].mean(0).reshape(shape1, shape2)
            )

        def process_hess(x, y):
            # mu mu0 pi rho
            hess_ = hess_fn(model_params_, x, y)
            mu_mu = get_hess_block(
                hess_, "mu", "mu", self.n_params_mu, self.n_params_mu
            )
            mu_mu0 = get_hess_block(
                hess_, "mu", "mu0", self.n_params_mu, self.n_params_mu0
            )
            mu_pi = get_hess_block(
                hess_, "mu", "pi", self.n_params_mu, self.n_params_pi
            )
            mu_rho = get_hess_block(
                hess_, "mu", "rho", self.n_params_mu, self.n_params_rho
            )
            mu0_mu0 = get_hess_block(
                hess_, "mu0", "mu0", self.n_params_mu0, self.n_params_mu0
            )
            mu0_pi = get_hess_block(
                hess_, "mu0", "pi", self.n_params_mu0, self.n_params_pi
            )
            mu0_rho = get_hess_block(
                hess_, "mu0", "rho", self.n_params_mu0, self.n_params_rho
            )
            pi_pi = get_hess_block(
                hess_, "pi", "pi", self.n_params_pi, self.n_params_pi
            )
            pi_rho = get_hess_block(
                hess_, "pi", "rho", self.n_params_pi, self.n_params_rho
            )
            rho_rho = get_hess_block(
                hess_, "rho", "rho", self.n_params_rho, self.n_params_rho
            )

            blk = jnp.block(
                [
                    [mu_mu, mu_mu0, mu_pi, mu_rho],
                    [mu_mu0.T, mu0_mu0, mu0_pi, mu0_rho],
                    [mu_pi.T, mu0_pi.T, pi_pi, pi_rho],
                    [mu_rho.T, mu0_rho.T, pi_rho.T, rho_rho],
                ]
            )
            return blk

        hessian = np.zeros((self.n_params, self.n_params), dtype=np.float64)
        for obs_id in tqdm(obs_ids, desc="Hessian computation"):
            x_ = jnp.array(x[[obs_id]], dtype=jnp.float64)
            y_ = jnp.array(y[[obs_id]], dtype=jnp.int32)
            x_obs = jax.device_put(x_, device)
            y_obs = jax.device_put(y_, device)
            hess_ = process_hess(x_obs, y_obs)
            hessian += hess_ / float(n_obs)
        return hessian
