"""Radial velocity exoplanet new module"""

import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import arviz as az

def exoplanet_MCMC(t, rv, rv_err, lit_period):

    # Initialize pymc3 model object:
    with pm.Model() as model:

        # Parameters
        logK = pm.Uniform(
            "logK",
            lower=0,
            upper=np.log(200),
            testval=np.log(0.5 * (np.max(rv) - np.min(rv))),
        )
        logP = pm.Uniform(
            "logP", lower=0, upper=np.log(5), testval=np.log(lit_period)
        )
        phi = pm.Uniform("phi", lower=0, upper=2 * np.pi, testval=0.1)

        # Parameterize the eccentricity using:
        hk = pmx.UnitDisk("hk", testval=np.array([0.01, 0.01]))
        e = pm.Deterministic("e", hk[0] ** 2 + hk[1] ** 2)
        w = pm.Deterministic("w", tt.arctan2(hk[1], hk[0]))

        rv0 = pm.Normal("rv0", mu=0.0, sd=10.0, testval=0.0)
        rvtrend = pm.Normal("rvtrend", mu=0.0, sd=10.0, testval=0.0)

        # Deterministic transformations
        n = 2 * np.pi * tt.exp(-logP)
        P = pm.Deterministic("P", tt.exp(logP))
        K = pm.Deterministic("K", tt.exp(logK))
        cosw = tt.cos(w)
        sinw = tt.sin(w)
        t0 = (phi + w) / n

        # The RV model
        bkg = pm.Deterministic("bkg", rv0 + rvtrend * t / 365.25)
        M = n * t - (phi + w)

        # This is the line that uses the custom Kepler solver
        f = xo.orbits.get_true_anomaly(M, e + tt.zeros_like(M))
        rvmodel = pm.Deterministic(
            "rvmodel", bkg + K * (cosw * (tt.cos(f) + e) - sinw * tt.sin(f))
        )

        # Condition on the observations
        pm.Normal("obs", mu=rvmodel, sd=rv_err, observed=rv)

        # Compute the phased RV signal
        phase = np.linspace(0, 1, 500)
        M_pred = 2 * np.pi * phase - (phi + w)
        f_pred = xo.orbits.get_true_anomaly(M_pred, e + tt.zeros_like(M_pred))
        rvphase = pm.Deterministic(
            "rvphase", K * (cosw * (tt.cos(f_pred) + e) - sinw * tt.sin(f_pred))
        )

    with model:
        map_params = pmx.optimize()

    with model:
        trace = pmx.sample(
            draws=1000,
            tune=1000,
            start=map_params,
            chains=2,
            cores=2,
            target_accept=0.95,
            return_inferencedata=True,
        )

    az.summary(
        trace,
        var_names=["logK", "logP", "phi", "e", "w", "rv0", "rvtrend"],
    )

    return trace, map_params, phase, rvphase
