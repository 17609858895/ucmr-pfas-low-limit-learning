"""Censoring-aware learners for left-censored concentrations (log10 ng/L scale).

Every model exposes .fit(X, y_lo, y_hi) and .predict_dist(X) -> dict with
  'cdf'(t): P(Y <= t) evaluated at log10 thresholds and 'logpdf'(y) for exact values.
y_lo == y_hi -> detected; y_lo == -inf -> left-censored at y_hi (the MRL).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats, optimize
from sklearn.linear_model import LogisticRegression

LN10 = np.log(10)


# ------------------------------------------------------------------ helpers
class Dist:
    """Mixture of a statistical point mass at zero (prob 1-pi) and location-scale family on log10 C."""

    def __init__(self, mu, sigma, pi=None, family="normal"):
        self.mu, self.sigma = np.asarray(mu, float), np.broadcast_to(np.asarray(sigma, float), np.shape(mu))
        self.pi = np.ones_like(self.mu) if pi is None else np.asarray(pi, float)
        self.fam = {"normal": stats.norm, "logistic": stats.logistic, "extreme": stats.gumbel_l}[family]

    def sf(self, t):  # P(Y > t) ; absent mass counts as below any threshold
        return self.pi * self.fam.sf((t - self.mu) / self.sigma)

    def cdf(self, t):
        return 1 - self.sf(t)

    def logpdf(self, y):
        return np.log(np.clip(self.pi, 1e-12, None)) + self.fam.logpdf((y - self.mu) / self.sigma) - np.log(self.sigma)

    def median(self):
        return np.where(self.pi > 0.5, self.mu + self.sigma * self.fam.ppf(np.clip(1 - 0.5 / self.pi, 1e-9, 1 - 1e-9)), -np.inf)


def censored_loglik(dist, y_lo, y_hi):
    det = np.isfinite(y_lo)
    ll = np.empty(len(y_hi))
    ll[det] = dist_sub(dist, det).logpdf(y_hi[det])
    ll[~det] = np.log(np.clip(dist_sub(dist, ~det).cdf(y_hi[~det]), 1e-12, None))
    return ll


def dist_sub(d, m):
    out = Dist.__new__(Dist)
    out.mu, out.sigma, out.pi, out.fam = d.mu[m], d.sigma[m], d.pi[m], d.fam
    return out


class Mixture:
    """Linear pool of Dist objects (stacking output)."""

    def __init__(self, dists, w):
        self.dists, self.w = dists, np.asarray(w, float)

    def sf(self, t):
        return sum(w * d.sf(t) for w, d in zip(self.w, self.dists))

    def cdf(self, t):
        return 1 - self.sf(t)

    def logpdf(self, y):
        return np.log(np.clip(sum(w * np.exp(d.logpdf(y)) for w, d in zip(self.w, self.dists)), 1e-300, None))


def mixture_loglik(mix, y_lo, y_hi):
    det = np.isfinite(y_lo)
    ll = np.empty(len(y_hi))
    ll[det] = np.log(np.clip(sum(w * np.exp(dist_sub(d, det).logpdf(y_hi[det])) for w, d in zip(mix.w, mix.dists)), 1e-300, None))
    ll[~det] = np.log(np.clip(sum(w * dist_sub(d, ~det).cdf(y_hi[~det]) for w, d in zip(mix.w, mix.dists)), 1e-12, None))
    return ll


# ------------------------------------------------------------------ baselines
class LGBClassifier:
    """Single-threshold classifier; only exposes P(Y > t0)."""

    def __init__(self, t0, **kw):
        self.t0 = t0
        self.kw = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)
        self.kw.update(kw)

    def fit(self, X, y_lo, y_hi):
        self.m = lgb.LGBMClassifier(**self.kw).fit(X, (np.isfinite(y_lo) & (y_hi >= self.t0)).astype(int))
        return self

    def prob_exceed(self, X):
        return self.m.predict_proba(X)[:, 1]


class LGBSubstitution:
    """Regression on log10 C with non-detects replaced by MRL/2; Gaussian residual for distribution."""

    def __init__(self, **kw):
        self.kw = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
        self.kw.update(kw)

    def fit(self, X, y_lo, y_hi):
        y = np.where(np.isfinite(y_lo), y_hi, y_hi - np.log10(2))
        self.m = lgb.LGBMRegressor(**self.kw).fit(X, y)
        self.s = np.std(y - self.m.predict(X)) * 1.15  # mild inflation for in-sample optimism
        return self

    def predict_dist(self, X):
        return Dist(self.m.predict(X), self.s)


class Tobit:
    """Linear censored-normal regression (ridge-penalised) on a preprocessed design matrix."""

    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, Z, y_lo, y_hi):
        det = np.isfinite(y_lo)
        Z1 = np.c_[np.ones(len(Z)), Z]

        def nll(p):
            b, ls = p[:-1], p[-1]
            s = np.exp(ls)
            mu = Z1 @ b
            zd = (y_hi[det] - mu[det]) / s
            zc = (y_hi[~det] - mu[~det]) / s
            ll = (stats.norm.logpdf(zd) - ls).sum() + stats.norm.logcdf(zc).sum()
            g_mu = np.zeros(len(mu))
            g_mu[det] = zd / s
            g_mu[~det] = -np.exp(stats.norm.logpdf(zc) - stats.norm.logcdf(zc)) / s
            gb = Z1.T @ g_mu
            g_ls = (zd ** 2 - 1).sum() + (-np.exp(stats.norm.logpdf(zc) - stats.norm.logcdf(zc)) * (-zc)).sum() * -1
            pen = 0.5 * self.l2 * (b[1:] ** 2).sum()
            gb[1:] -= self.l2 * b[1:]
            return -(ll - pen), -np.r_[gb, g_ls]

        p0 = np.r_[np.median(y_hi), np.zeros(Z.shape[1]), np.log(0.5)]
        r = optimize.minimize(nll, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        self.b, self.s = r.x[:-1], np.exp(r.x[-1])
        return self

    def predict_dist(self, Z):
        return Dist(np.c_[np.ones(len(Z)), Z] @ self.b, self.s)


# ------------------------------------------------------------------ survival ML
class XGBAFT:
    """Interval-censored accelerated failure time gradient boosting (Barnwal et al. 2022).

    Concentration plays the role of survival time; non-detects are left-censored at the MRL.
    """

    def __init__(self, family="normal", sigma=0.6, rounds=800, **kw):
        self.family, self.sigma, self.rounds = family, sigma, rounds
        self.p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution=family,
                      aft_loss_distribution_scale=sigma, tree_method="hist", learning_rate=0.03,
                      max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                      reg_lambda=1.0, nthread=16)
        self.p.update(kw)

    def _dm(self, X, y_lo=None, y_hi=None, w=None):
        dm = xgb.DMatrix(X, enable_categorical=True, weight=w)
        if y_lo is not None:
            # xgboost works on the natural time scale; convert log10 ng/L back to ng/L
            lo = np.where(np.isfinite(y_lo), 10 ** y_lo, 0.0)
            dm.set_float_info("label_lower_bound", lo)
            dm.set_float_info("label_upper_bound", 10 ** y_hi)
        return dm

    def fit(self, X, y_lo, y_hi, w=None):
        self.b = xgb.train(self.p, self._dm(X, y_lo, y_hi, w), num_boost_round=self.rounds)
        return self

    def predict_dist(self, X):
        mu_ln = np.log(self.b.predict(self._dm(X), output_margin=True))  # margin is exp(mu)? see below
        return Dist(mu_ln, self.sigma / LN10, family=self.family)

    def mu_log10(self, X):
        # output_margin=True returns mu on natural-log time scale
        return self.b.predict(self._dm(X), output_margin=True) / LN10


def _xgbaft_predict_dist(self, X):
    return Dist(self.mu_log10(X), self.sigma / LN10, family=self.family)


XGBAFT.predict_dist = _xgbaft_predict_dist


class CureAFT:
    """Mixture-cure survival boosting for left-censored concentrations (proposed).

    Y = zero-component membership with prob 1-pi(x);
    otherwise log C ~ AFT(mu(x), sigma).  Component membership is a statistical parameter, not proof of PFAS absence.
    A single AFT represents the response with one continuous tail.
    Fitted by EM: E-step responsibilities for censored rows, M-step a weighted LightGBM incidence
    model and a weighted XGBoost-AFT latency model.
    """

    def __init__(self, family="normal", sigma=0.6, n_iter=6, rounds=500, clf_kw=None, aft_kw=None):
        self.family, self.sigma, self.n_iter, self.rounds = family, sigma, n_iter, rounds
        self.clf_kw = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
        self.clf_kw.update(clf_kw or {})
        self.aft_kw = aft_kw or {}

    def fit(self, X, y_lo, y_hi):
        det = np.isfinite(y_lo)
        fam = Dist([0.0], 1.0, family=self.family).fam
        w = np.where(det, 1.0, 0.5)  # initial responsibility of 'impacted'
        Xd = pd.concat([X, X], ignore_index=True)
        lab = np.r_[np.ones(len(X)), np.zeros(len(X))]
        for it in range(self.n_iter):
            # M-step incidence: soft labels via duplicated rows
            self.clf = lgb.LGBMClassifier(**self.clf_kw).fit(Xd, lab, sample_weight=np.r_[w, 1 - w] + 1e-6)
            # M-step latency: AFT weighted by responsibility
            self.aft = XGBAFT(self.family, self.sigma, rounds=self.rounds, **self.aft_kw).fit(X, y_lo, y_hi, w=w + 1e-6)
            pi = np.clip(self.clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
            F = fam.cdf((y_hi - self.aft.mu_log10(X)) / (self.sigma / LN10))
            w = np.where(det, 1.0, pi * F / ((1 - pi) + pi * F))
        return self

    def predict_dist(self, X):
        pi = self.clf.predict_proba(X)[:, 1]
        return Dist(self.aft.mu_log10(X), self.sigma / LN10, pi=pi, family=self.family)


def fit_stack_weights(oof_dists, y_lo, y_hi):
    """Linear-pool stacking weights maximising out-of-fold censored log score (simplex)."""
    det = np.isfinite(y_lo)
    comp = []
    for d in oof_dists:
        c = np.empty(len(y_hi))
        c[det] = np.exp(dist_sub(d, det).logpdf(y_hi[det]))
        c[~det] = dist_sub(d, ~det).cdf(y_hi[~det])
        comp.append(np.clip(c, 1e-300, None))
    C = np.column_stack(comp)

    def f(a):
        w = np.exp(a - a.max()); w /= w.sum()
        return -np.log(C @ w + 1e-300).mean()

    r = optimize.minimize(f, np.zeros(C.shape[1]), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
    w = np.exp(r.x - r.x.max())
    return w / w.sum()
