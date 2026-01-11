import os
from typing import Union

import flax.linen as nn
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.distributions import Categorical, Normal, Poisson
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ._template import PPIAbstractClass
from ._utils import _zstat_generic2, optimize_ppi, optimize_ppi_gd


class PoissonModelIntercept(nn.Module):
    n_classes: int
    n_features: int
    mu_prior_std: Union[float, jnp.ndarray]
    n_obs_real: int

    def setup(self):
        self.pi_raw = self.param("pi", nn.initializers.normal(), (self.n_classes - 1,))
        self.mu0 = self.param("mu0", nn.initializers.normal(), (self.n_features))
        self.mu = self.param(
            "mu", nn.initializers.normal(), (self.n_classes - 1, self.n_features)
        )

    def __call__(self, x, y):
        y_ = y.astype(jnp.int32)
        pi = jnp.concatenate([self.pi_raw, -jnp.sum(self.pi_raw, keepdims=True)])
        mu_placeholder = jnp.zeros_like(self.mu0)
        mu = jnp.concatenate([mu_placeholder[None], self.mu], axis=0)
        y_oh = jnp.eye(self.n_classes)[y_]
        mus_ = y_oh @ mu + self.mu0
        rates = jnp.exp(mus_)

        # Categorical
        log_pc = Categorical(logits=pi).log_prob(y_)
        log_px_c = Poisson(rate=rates).log_prob(x).sum(axis=-1)

        # Regularization & priors
        # reg1 = -jnp.sum(self.pi_raw**2)
        reg1 = 0.0
        if self.mu_prior_std is not None:
            reg2 = (
                1.0
                / self.n_obs_real
                * Normal(loc=0.0, scale=self.mu_prior_std).log_prob(self.mu).sum()
            )
        else:
            reg2 = 0.0
        priors = reg1 + reg2

        loss = -(log_pc + log_px_c + priors)
        return {
            "loss": loss,
            "log_pc": log_pc,
            "log_px_c": log_px_c,
        }


class PoissonIntercept(PPIAbstractClass):
    def __init__(
        self,
        mu_prior_std=None,
        weight_init="smart",
        optimizer="gd",
        optimizer_kwargs=None,
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
        self.n_params = (
            (self.n_classes - 1) * self.n_features
            + self.n_features
            + (self.n_classes - 1)
        )
        self.model = PoissonModelIntercept(
            n_classes=self.n_classes,
            n_features=self.n_features,
            mu_prior_std=mu_prior_std,
            n_obs_real=n_obs_real,
        )
        self.model_params = None
        if weight_init == "smart":
            self.smart_init()
        elif weight_init == "zero":
            self.zero_init()
        self.lambd_ = None
        self.log = None
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

    def fit(self, lambd_=None):
        if lambd_ is None:
            lambd_ = self.get_lambda()
            self.lambd_ = lambd_

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
        return np.hstack([mu, mu0, pi])

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
            all_grads[i : i + batch_size] = np.hstack([grad_mu, grad_mu0, grad_pi])
        return np.array(all_grads)

    def _construct_contrast(self, feature_id, idx_a):
        mu_contrast = np.zeros((self.n_classes - 1, self.n_features))
        mu_contrast[idx_a - 1, feature_id] = 1.0

        mu0_contrast = np.zeros(self.n_features)
        pi_contrast = np.zeros(self.n_classes - 1)
        contrast = np.hstack([mu_contrast.flatten(), mu0_contrast, pi_contrast])
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

    def _get_param_id(self, feature_id=None, class_id=None, param_type=None):
        n_params_mu = self.n_features * (self.n_classes - 1)
        n_params_mu0 = self.n_features
        if param_type == "mu":
            return ((class_id - 1) * self.n_features) + feature_id
        elif param_type == "mu0":
            return n_params_mu + feature_id
        elif param_type == "pi":
            return n_params_mu + n_params_mu0 + (class_id - 1)

    def _get_param_mask(self, feature_id):
        mu_indices = [
            self._get_param_id(
                feature_id=feature_id, class_id=class_id, param_type="mu"
            )
            for class_id in range(1, self.n_classes)
        ]
        mu0_indices = [self._get_param_id(feature_id=feature_id, param_type="mu0")]
        pi_indices = [
            self._get_param_id(class_id=class_id, param_type="pi")
            for class_id in range(1, self.n_classes)
        ]
        indices_to_keep = np.hstack([mu_indices, mu0_indices, pi_indices])
        return indices_to_keep

    def differential_expression_ew(
        self, idx_a, feature_names=None, filter_pval_thresh=3e-2, mode="overall"
    ):
        idx_a_ = idx_a - 1
        results = []
        for feature_id in range(self.n_features):
            mask_ = self._get_param_mask(feature_id)
            v_ = self.v[mask_][:, mask_]
            hess_ = self.hessian[mask_][:, mask_]

            beta = self.theta[mask_]
            cov = self._compute_sigma(hess_, v_, self.n, mode=mode)

            is_hess_below_thresh = hess_[idx_a_, idx_a_] < filter_pval_thresh
            if is_hess_below_thresh:
                pval = 1.0
            else:
                _, pval = _zstat_generic2(
                    beta[idx_a_], np.sqrt(cov[idx_a_, idx_a_]), alternative="two-sided"
                )
            results.append(
                {
                    "pval": pval,
                    "hess": hess_[idx_a_, idx_a_],
                    "beta": beta[idx_a_],
                    "cov": cov[idx_a_, idx_a_],
                }
            )
        res = pd.DataFrame(results)
        res["pval"].iloc[np.isnan(res["pval"])] = 1.0
        res["padj"] = multipletests(res["pval"], method="fdr_bh")[1]
        res["is_significant_005"] = res["padj"] < 0.05
        if feature_names is not None:
            res["feature_name"] = feature_names
        return res

    def smart_init(self):
        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl

        y_all = np.hstack([y_gt, y_hat, y_unl])
        x_all_ = np.vstack([x_gt, x_hat, x_unl])
        x_all_ = np.log1p(x_all_)
        n_total = y_all.shape[0]

        y_all_oh = np.eye(self.n_classes)[y_all]
        probs_hat = y_all_oh.sum(0) / n_total
        probs_hat = np.log(probs_hat + 1e-6)
        pi0 = probs_hat[1:] - probs_hat[0]

        design_ = y_all_oh.copy()
        design_[:, 0] = 1.0

        mu_all = np.linalg.pinv(design_.T @ design_) @ design_.T @ x_all_
        mu = mu_all[1:]
        mu0 = mu_all[0]
        params = {
            "params": {
                "mu": jnp.array(mu, dtype=jnp.float64),
                "mu0": jnp.array(mu0, dtype=jnp.float64),
                "pi": jnp.array(pi0, dtype=jnp.float64),
            }
        }
        self.model_params = params

    def zero_init(self):
        mu = np.zeros((self.n_classes - 1, self.n_features))
        mu0 = np.zeros(self.n_features)
        pi0 = np.zeros(self.n_classes - 1)
        params = {
            "params": {
                "mu": jnp.array(mu, dtype=jnp.float64),
                "mu0": jnp.array(mu0, dtype=jnp.float64),
                "pi": jnp.array(pi0, dtype=jnp.float64),
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

        def process_hess(x, y):
            hess_ = hess_fn(model_params_, x, y)
            mu_mu = (
                hess_["params"]["mu"]["params"]["mu"]
                .mean(0)
                .reshape(
                    (self.n_classes - 1) * self.n_features,
                    (self.n_classes - 1) * self.n_features,
                )
            )
            mu_mu0 = (
                hess_["params"]["mu"]["params"]["mu0"]
                .mean(0)
                .reshape((self.n_classes - 1) * self.n_features, self.n_features)
            )
            mu_pi = (
                hess_["params"]["mu"]["params"]["pi"]
                .mean(0)
                .reshape((self.n_classes - 1) * self.n_features, self.n_classes - 1)
            )
            mu0_mu0 = (
                hess_["params"]["mu0"]["params"]["mu0"]
                .mean(0)
                .reshape(self.n_features, self.n_features)
            )
            mu0_pi = (
                hess_["params"]["mu0"]["params"]["pi"]
                .mean(0)
                .reshape(self.n_features, self.n_classes - 1)
            )
            pi_pi = (
                hess_["params"]["pi"]["params"]["pi"]
                .mean(0)
                .reshape(self.n_classes - 1, self.n_classes - 1)
            )
            blk = jnp.block(
                [
                    [mu_mu, mu_mu0, mu_pi],
                    [mu_mu0.T, mu0_mu0, mu0_pi],
                    [mu_pi.T, mu0_pi.T, pi_pi],
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
