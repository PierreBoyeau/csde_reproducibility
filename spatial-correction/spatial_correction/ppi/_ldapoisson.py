import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions import Categorical, Poisson

from ._utils import optimize_ppi

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from sklearn.preprocessing import LabelEncoder

from ._template import PPIAbstractClass


class PoissonLDAModel(nn.Module):
    n_classes: int
    n_features: int

    def setup(self):
        self.pi_raw = self.param("pi", nn.initializers.normal(), (self.n_classes - 1,))
        self.mu = self.param(
            "mu", nn.initializers.normal(), (self.n_classes, self.n_features)
        )

    def __call__(self, x, y):
        y_ = y.astype(jnp.int32)
        pi = jnp.concatenate([self.pi_raw, -jnp.sum(self.pi_raw, keepdims=True)])
        log_pc = Categorical(logits=pi).log_prob(y_)
        rates = jnp.exp(self.mu[y_]) + 1e-6
        log_px_c = Poisson(rate=rates).log_prob(x).sum(axis=-1)
        loss = -(log_pc + log_px_c)
        reg1 = jnp.sum(self.pi_raw**2)
        loss = loss + reg1
        return {
            "loss": loss,
            "log_pc": log_pc,
            "log_px_c": log_px_c,
        }


class PPIPoissonLDA(PPIAbstractClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl
        all_y = np.hstack([y_gt, y_hat, y_unl])
        self.le = LabelEncoder()
        self.le.fit(all_y)
        y_gt = self.le.transform(y_gt)
        y_hat = self.le.transform(y_hat)
        y_unl = self.le.transform(y_unl)
        self.inputs_gt = (x_gt, y_gt)
        self.inputs_hat = (x_hat, y_hat)
        self.inputs_unl = (x_unl, y_unl)

        self.n_features = x_gt.shape[1]
        self.n_classes = self.le.classes_.shape[0]
        self.model = PoissonLDAModel(
            n_classes=self.n_classes, n_features=self.n_features
        )
        self.model_params = None
        self.lambd_ = None
        self.log = None

    def fit(self, lambd_=None):
        if lambd_ is None:
            lambd_ = self.get_lambda()
            self.lambd_ = lambd_
        self.get_asymptotic_distribution(lambd_)

    def get_pointestimate(self, lambd_, **kwargs):
        x_gt, y_gt = self.inputs_gt
        x_hat, y_hat = self.inputs_hat
        x_unl, y_unl = self.inputs_unl
        model_params = optimize_ppi(
            self.model,
            lambd_=lambd_,
            x_gt=x_gt,
            y_gt=y_gt,
            x_hat=x_hat,
            y_hat=y_hat,
            x_unl=x_unl,
            y_unl=y_unl,
            **kwargs,
        )
        self.model_params = model_params

        mu = np.array(model_params["params"]["mu"].reshape(-1))
        pi = np.array(model_params["params"]["pi"])
        return np.hstack([mu, pi])

    def grad_fn(self, inputs):
        x, y = inputs
        n_obs = x.shape[0]

        @jax.jit
        def likelihood(model_params, x, y):
            return self.model.apply(model_params, x, y)["loss"]

        score = jax.jit(jax.jacfwd(likelihood))
        grads = score(self.model_params, x, y)

        grad_mu = grads["params"]["mu"].reshape(n_obs, -1)
        grad_pi = grads["params"]["pi"]
        all_grads = jnp.hstack([grad_mu, grad_pi])
        return np.array(all_grads)

    def _construct_contrast(self, feature_id, idx_a, idx_b=None):
        mu_contrast = np.zeros((self.n_classes, self.n_features))
        mu_contrast[idx_a, feature_id] = 1.0

        idx_b = (
            np.array([idx_b])
            if idx_b is not None
            else np.setdiff1d(np.arange(self.n_classes), idx_a)
        )
        n_b = len(idx_b)
        mu_contrast[idx_b, feature_id] = -1.0 / n_b

        pi_contrast = np.zeros(self.n_classes - 1)
        contrast = np.hstack([mu_contrast.flatten(), pi_contrast])
        return contrast

    def construct_contrast(self, idx_a, idx_b=None):
        _contrast = [
            self._construct_contrast(feature_id, idx_a, idx_b)
            for feature_id in range(self.n_features)
        ]
        _contrast = np.vstack(_contrast)
        return _contrast

    def get_beta(self, class_a, class_b=None):
        idx_a = self.le.transform([class_a])[0]
        idx_b = self.le.transform([class_b])[0] if class_b is not None else None

        contrast = self.construct_contrast(idx_a, idx_b)
        beta = contrast @ self.theta
        cov = contrast @ self.sigma @ contrast.T
        return beta, cov

    def hessian_fn(self, inputs):
        x, y = inputs

        @jax.jit
        def likelihood(model_params, x, y):
            return self.model.apply(model_params, x, y)["loss"]

        score = jax.jacfwd(likelihood)
        hess_fn = jax.jit(jax.jacfwd(score))

        hess_ = hess_fn(self.model_params, x, y)
        mu_mu = np.array(hess_["params"]["mu"]["params"]["mu"].mean(0)).reshape(
            self.n_classes * self.n_features, self.n_classes * self.n_features
        )
        mu_pi = np.array(hess_["params"]["mu"]["params"]["pi"].mean(0)).reshape(
            self.n_classes * self.n_features, self.n_classes - 1
        )
        pi_pi = np.array(hess_["params"]["pi"]["params"]["pi"].mean(0))

        block_mat = np.block(
            [
                [mu_mu, mu_pi],
                [mu_pi.T, pi_pi],
            ]
        )
        return block_mat
