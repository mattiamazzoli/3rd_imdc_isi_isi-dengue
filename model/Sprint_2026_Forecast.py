#!/usr/bin/env python
# coding: utf-8

# ISI Dengue Model
# ==============================================
# 
# This code implements a comprehensive dengue transmission model for Brazilian States. The model combines:
# 
# 1. A compartmental ODE system modeling dengue transmission between vectors (Aedes aegypti) 
#    and humans
# 2. Weather-dependent parameters affecting vector biology
# 3. Bayesian parameter estimation using PyMC
# 4. Forecasting capabilities with uncertainty quantification
# 5. Integration with epidemiological surveillance data

#  - Validation test 1: Predict the weekly number of dengue or chikungunya cases for the 2022–2023 season (EW41 2022 – EW40 2023), using data from EW01 2010 to EW25 2022.  
# - Validation test 2: Predict the weekly number of dengue or chikungunya cases for the 2023–2024 season (EW41 2023 – EW40 2024), using data from EW01 2010 to EW25 2023.  
# - Validation test 3: Predict the weekly number of dengue or chikungunya cases for the 2024–2025 season (EW41 2024 – EW40 2025), using data from EW01 2010 to EW25 2024.  
# - Validation test 4: Predict the weekly number of dengue or chikungunya cases for the 2025–2026 season (EW41 2025 – EW40 2026), using data from EW01 2010 to EW25 2025.  

# In[1]:


import numpy as np
from utils import *
from upload_utils import *
from model_utils import *
from forecast_plot import *
from datetime import datetime, timedelta
import pandas as pd
import time
from numba import njit
from sympy import exp  
import warnings
warnings.filterwarnings('ignore')
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import scipy.stats as st
import os, pickle, json
from datetime import datetime
import geopandas as gpd


import aiohttp
import asyncio

from mosqlient import upload_prediction, validate_prediction, delete_prediction


# In[3]:


from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
#import requests
from numba import njit  
from concurrent.futures import ThreadPoolExecutor, as_completed


# In[4]:


import arviz as az  
import corner  
import traceback
from scipy.stats import norm  # pyright: ignore[reportMissingImports]
from scipy.stats import beta, uniform # pyright: ignore[reportMissingImports]
from pathlib import Path
import string
from typing import Any
import xarray as xr 



validation_round = 1
first_date = '2019-09-22' #'2020-10-05' #'2019-10-07' #'2020-10-05'
if validation_round == 1:
    train_start_date = first_date
    train_end_date = '2022-06-19'

    forecast_start_date = '2022-10-09'
    forecast_end_date = '2023-10-01'

elif validation_round == 2:
    train_start_date = first_date
    train_end_date = '2023-06-18'

    forecast_start_date = '2023-10-08'
    forecast_end_date = '2024-09-29'

elif validation_round == 3:
    train_start_date = first_date
    train_end_date = '2024-06-16'

    forecast_start_date = '2024-10-06'
    forecast_end_date = '2025-09-28'

elif validation_round == 4:
    train_start_date = first_date
    train_end_date = '2025-06-15'

    forecast_start_date = '2025-10-05'
    forecast_end_date = '2026-10-04'



# State administrative mapping for competition submission
adm_1_map = {
    'Acre': 'AC', 'Alagoas': 'AL', 'Amapá': 'AP', 'Amazonas': 'AM', 'Bahia': 'BA', 'Ceará': 'CE',
    'Distrito Federal': 'DF', 'Goiás': 'GO', 'Maranhão': 'MA', 'Mato Grosso': 'MT',
    'Mato Grosso do Sul': 'MS', 'Minas Gerais': 'MG', 'Pará': 'PA', 'Paraíba': 'PB',
    'Paraná': 'PR', 'Pernambuco': 'PE', 'Piauí': 'PI', 'Rio de Janeiro': 'RJ', 'Rio Grande do Norte': 'RN',
    'Rio Grande do Sul': 'RS', 'Rondônia': 'RO', 'Roraima': 'RR', 'Santa Catarina': 'SC',
    'São Paulo': 'SP', 'Sergipe': 'SE', 'Tocantins': 'TO'
}

# Complete list of Brazilian states for analysis ()
brazilian_states = [
    'Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará',
    'Distrito Federal', 'Goiás', 'Maranhão', 'Mato Grosso',
    'Mato Grosso do Sul', 'Minas Gerais', 'Pará', 'Paraíba',
    'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro', 'Rio Grande do Norte',
    'Rio Grande do Sul', 'Rondônia', 'Roraima', 'Santa Catarina',
    'São Paulo', 'Sergipe', 'Tocantins'
]


state_code = pd.read_csv('./data_imdc_2026/map_regional_health.csv')
state_code = dict(zip(state_code.uf.values, state_code.uf_code.values))
len(brazilian_states)


geo_data = gpd.read_file('./data_imdc_2026/shape_muni.gpkg')
geo_data.head()


# ── Resource control (run this FIRST, before heavy imports do work) ──
# Cap CPU usage to 6 of 8 cores and stop BLAS/OpenMP/numba from
# over-subscribing threads inside each worker process. Without these
# caps, 8 SMC workers each spawn their own BLAS thread pool -> dozens
# of threads + duplicated RAM. Setting them to 1 keeps each worker
# single-threaded so the *process* pool is the only parallelism.
import os

N_CORES = 8  # use 6 of 8 physical cores; leaves headroom for the OS/UI

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
# numba: keep each simulate_dengue_fast call single-threaded so it does
# not fight the SMC process pool for cores.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass



@as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar], otypes=[pt.dvector])
def dengue_sim_fast(k_v, k_h, s_0, b_factor, inc_factor):

    params = {
        "k_v": float(k_v), 
        "k_h": float(k_h), 
        "s_0": float(s_0), 
        'b_factor': float(b_factor), 
        'inc_factor': float(inc_factor)
    }

    try:
        weekly_cases, Nv, Nh = simulate_dengue_wrapper(
            params, 
            sim_data.egg_laying_rate, 
            sim_data.egg_development_rate,
            sim_data.bite_rate, 
            sim_data.inc_rate, 
            sim_data.initial_cases_df,
            sim_data.initial_vectors_df,
            sim_data.days
        )

        # Store full simulation for later analysis
        sim_data.full_weekly_cases = weekly_cases
        sim_data.full_Nv = Nv
        sim_data.full_Nh = Nh

        # Only use fit weeks for likelihood
        if sim_data.fit_weeks is not None:
            weekly_cases = weekly_cases[sim_data.fit_weeks]

        return weekly_cases
    except:
        if sim_data.fit_weeks is not None:
            return np.full(len(sim_data.fit_weeks), 1e-6)
        else:
            return np.full(len(sim_data.csv_state_cases_df), 1e-6)



#Forecasting and Prediction
#==========================
#Forecasting and Prediction
#==========================
def forecast_function(state, forecast_start_date, forecast_end_date, metadata, posterior_data,
                     train_start_date, train_end_date, geo_data):
    """
    Generate probabilistic dengue forecasts using trained model.
    
    Uses posterior parameter samples to generate ensemble forecasts with
    full uncertainty quantification. Handles weather data extrapolation
    for forecasts beyond available meteorological data.
    
    Parameters:
    -----------
    state : str
        Target state for forecasting
    forecast_start_date : str
        Forecast period start (YYYY-MM-DD), repeat the simulation from train_start_date and extract the forecasted period
    forecast_end_date : str
        Forecast period end (YYYY-MM-DD)
    metadata : dict
        Model configuration from training
    posterior_data : dict
        Parameter samples from Bayesian fitting
        
    Returns:
    --------
    forecast_df : pd.DataFrame
        Probabilistic forecasts with confidence intervals
    control_cases : pd.DataFrame
        Observed data for forecast validation
    """

    forecast_weather_end_date = forecast_end_date #use this to fetch weather forecast data
    
    if forecast_end_date > '2026-03-15':
        forecast_end_date = '2026-03-15'         #use this to catch cases data
         
    # Parse forecast period
    forecast_start = datetime.strptime(forecast_start_date, "%Y-%m-%d")
    forecast_end = datetime.strptime(forecast_end_date, "%Y-%m-%d")
    forecast_weather_end = datetime.strptime(forecast_weather_end_date, "%Y-%m-%d")

    date_difference = forecast_weather_end - forecast_start
    forecast_days = date_difference.days

    # Align forecast period with weekly reporting 
    if forecast_days % 7 != 0:
        forecast_end = forecast_weather_end - timedelta(days=forecast_days % 7) + timedelta(days=7)
        forecast_days = (forecast_weather_end - forecast_start).days
    
    # Collect geographic and population data
    geo_data_state = geo_data[geo_data['uf'] == state]
    state_geocodes = geo_data_state['geocode'].astype(int).tolist()
    
    # Get validation data (observed cases during forecast period)
    control_cases, major_cities = fetch_dengue_data_state_from_csv(
        state_geocodes,
        pd.to_datetime(forecast_start_date),
        pd.to_datetime(forecast_end_date)
    )
    
    # Use training data for initial conditions 
    # Use previous_week_date to align with fit function
    previous_week_date = str(datetime.date(datetime.strptime(train_start_date, "%Y-%m-%d") - timedelta(days=1)))
    train_cases, major_cities = fetch_dengue_data_state_from_csv(
        state_geocodes,
        pd.to_datetime(previous_week_date),
        pd.to_datetime(train_end_date)
    )
    
    # Use the average of the first and last 6 weeks of training data
    last_six_weeks = train_cases.tail(6)
    tail_avg_cases_6 = last_six_weeks['casos'].mean()
    first_six_weeks = train_cases.head(6)
    head_avg_cases_6 = first_six_weeks['casos'].mean()

    tail_head_per = abs(tail_avg_cases_6 - head_avg_cases_6) / min(tail_avg_cases_6, head_avg_cases_6)
    if tail_head_per > 5:
        if head_avg_cases_6 > tail_avg_cases_6:
            head_avg_cases_6 = head_avg_cases_6 / 3
        else:
            tail_avg_cases_6 = tail_avg_cases_6 / 3
            
    initial_cases = (head_avg_cases_6 + tail_avg_cases_6) / 2 / 7
    population = train_cases.iloc[-1]['pop']
    initial_cases_df = pd.DataFrame([[initial_cases, population]], columns=['casos','pop'])

    # Prepare initial conditions for forecast simulation 
    initial_vectors_df = load_or_fetch_vectors(state, geo_data_state, forecast_start_date, forecast_weather_end_date, 'forecast')
    print(initial_vectors_df.head())
    forecast_vectors_df = initial_vectors_df.iloc[[0], :]  
    
    # Get weather parameters from training metadata
    weather_coeffs = metadata['weather_coeffs']
    
    # Handle weather data availability (matches fit function approach)
    weather_start = str(datetime.date(datetime.strptime(previous_week_date, "%Y-%m-%d") - timedelta(days=2)))
    
    # FIX: Load weather data using the same function as fit function
    weather_data_full = load_or_fetch_weather(
        state, 
        weather_start, 
        forecast_weather_end_date, 
        weather_coeffs, 
        major_cities,
        'forecast',
    )
    
    # Extract weather-dependent biological parameters
    # FIX: Use the same indexing approach as fit_function
    inc_rate = np.array(weather_data_full['incubation'], dtype=np.float64)
    bite_rate = np.array(weather_data_full['bite'], dtype=np.float64)
    egg_laying_rate = np.array(weather_data_full['egg'], dtype=np.float64)
    egg_development_rate = np.array(weather_data_full['theta'], dtype=np.float64)
    

    # Generate ensemble forecasts using posterior samples.
    kv_s  = posterior_data['k_v_samples']
    kh_s  = posterior_data['k_h_samples']
    s0_s  = posterior_data['s_0_samples']
    bf_s  = posterior_data['b_factor_samples']
    inc_s = posterior_data['inc_factor_samples']
    n_total_samples = len(kv_s)
    forecast_results = []

    # Run forecast simulation for each posterior sample
    for i in range(n_total_samples):
        sample_params = {
            'k_v':        kv_s[i],
            'k_h':        kh_s[i],
            's_0':        s0_s[i],
            'b_factor':   bf_s[i],
            'inc_factor': inc_s[i],
        }

        weekly_forecast, Nv_forecast, Nh_forecast = simulate_dengue_wrapper(
            sample_params, egg_laying_rate, egg_development_rate,
            bite_rate, inc_rate, initial_cases_df, forecast_vectors_df, forecast_days
        )

        forecast_results.append(weekly_forecast)

    # Convert to array for statistical analysis
    forecast_results = np.array(forecast_results)
    
    # Calculate prediction intervals at multiple confidence levels
    confidence_levels = [50, 80, 90, 95]
    forecast_stats = {}
    for level in confidence_levels:
        lower_bound = 50 - level / 2
        upper_bound = 50 + level / 2
        forecast_stats[f'ci_{level}_lower'] = np.percentile(forecast_results, lower_bound, axis=0)
        forecast_stats[f'ci_{level}_upper'] = np.percentile(forecast_results, upper_bound, axis=0)

    # Calculate central tendency and dispersion
    forecast_stats['pred'] = np.median(forecast_results, axis=0)
    forecast_stats['std'] = np.std(forecast_results, axis=0)

    # Format forecast results
    forecast_df = pd.DataFrame({
        'week': range(len(forecast_stats['pred'])),
        'pred': forecast_stats['pred'],
        'std': forecast_stats['std']
    })
    
    # Add confidence intervals
    for level in confidence_levels:
        forecast_df[f'lower_{level}'] = forecast_stats[f'ci_{level}_lower']
        forecast_df[f'upper_{level}'] = forecast_stats[f'ci_{level}_upper']

    # Add temporal dimension
    forecast_dates = pd.date_range(start=forecast_start, freq='W', periods=len(forecast_stats['pred']))
    forecast_df['date'] = forecast_dates

    return forecast_df, control_cases



forecast_start = datetime.strptime(forecast_start_date, "%Y-%m-%d")
train_start = datetime.strptime(train_start_date, "%Y-%m-%d")

datdiff = forecast_start - train_start
print(datdiff.days)


# Main Execution Pipeline
# ======================

# Execute complete modeling pipeline for all Brazilian states.
# 
# This loop performs:
# 1. Bayesian parameter estimation using historical data
# 2. Probabilistic forecasting with uncertainty quantification  
# 3. Competitive submission to digital epidemiology platform
# 

# In[24]:


class SimData:
    """Stores data globally so the independent function can see it"""
    egg_laying_rate = None
    egg_development_rate = None
    bite_rate = None
    inc_rate = None
    initial_cases_df = None
    initial_vectors_df = None
    days = None
    csv_state_cases_df = None
    fit_weeks = None
    full_weekly_cases = None
    full_Nh = None
    full_Nv = None
    final_state_ensemble = None
    final_state_median = None

sim_data = SimData()


def simulate_dengue_wrapper(params, egg_lrate, egg_drate, bite_rate, inc_rate, 
                            cases_df, vectors_df, days):

    """
    Wrapper function for dengue simulation with data formatting.
    NOW HANDLES 12 COMPARTMENTS (with asymptomatic Ah)
    """

    k_v = params['k_v']
    k_h = params['k_h']
    s_0 = params['s_0']
    b_factor = params['b_factor']
    inc_factor = params['inc_factor']

    tot_cases = float(cases_df['casos'].iloc[0])
    tot_pop = float(cases_df['pop'].iloc[0])
    tot_vectors = float(vectors_df['Sv'].iloc[0])
    tot_eggs = float(vectors_df['Pv'].iloc[0])

    # Run core simulation (now returns 12 compartments)
    results, dt = simulate_dengue_fast(k_v, k_h, s_0, b_factor, inc_factor,
                                       tot_cases, tot_pop, tot_vectors, tot_eggs,
                                       egg_lrate, egg_drate, bite_rate, inc_rate,
                                       days)

    # Calculate weekly case incidence (symptomatic only)
    _, weekly_cases = calculate_weekly_cases(results, dt)

    # Extract population sizes for R0 calculation (UPDATED INDICES)
    Nv = results[-1, 1] + results[-1, 2] + results[-1, 4]  # Vector population
    Nh = results[-1, 6] + results[-1, 7] + results[-1, 8] + results[-1, 9] + results[-1, 10]  # Human population

    return weekly_cases, np.full(days, Nv), np.full(days, Nh)



def fit_function(state, start_date, end_date, geodata, progress_bar_bool=True, 
                alpha_fixed=0.915802, fit_weeks=None, plot_results=True,
                plot_priors=True, prior_config="informative"):
    """
    Fit dengue transmission model using Bayesian parameter estimation.

    Parameters:
    ----------- 
    state : str
        Brazilian state name for analysis
    start_date : str
        Training period start date (YYYY-MM-DD)
    end_date : str
        Training period end date (YYYY-MM-DD)
    progress_bar_bool : bool
        Show MCMC progress bar
    alpha_fixed : float
        Fixed alpha parameter for negative binomial likelihood
    fit_weeks : list, optional
        Specific weeks to fit (indices)
    plot_results : bool
        Whether to generate posterior plots

    Returns:
    --------
    best_fit_params : dict
        Maximum a posteriori parameter estimates
    trace : arviz.InferenceData
        Full posterior samples for further analysis

    """
    # Create unique timestamp identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    previous_week_date = str(datetime.date(datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)))
    weather_start_date = str(datetime.date(datetime.strptime(previous_week_date, "%Y-%m-%d") - timedelta(days=2)))
    date_difference = datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
    days = date_difference.days

    if days % 7 != 0:
        end_date = str(datetime.date(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days % 7) + timedelta(days=7)))
        date_difference = datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")
        days = date_difference.days

    geo_data_state = geo_data[geo_data['uf'] == state]
    state_geocodes = geo_data_state['geocode'].astype(int).tolist()

    csv_state_cases_df, major_cities = load_or_fetch_cases(
        state, state_geocodes, previous_week_date, end_date, validation_round
    )


    vectors_df = load_or_fetch_vectors(state, geo_data_state, start_date, end_date, 'train', validation_round)
    initial_cases_df = csv_state_cases_df.iloc[[0],:]
    print(initial_cases_df.head())
    print(vectors_df.head())
    initial_vectors_df = vectors_df.iloc[[0],:]

    weather_data_df = load_or_fetch_weather(state, weather_start_date, end_date, 
                                            dict_weather_coeffs, major_cities, 'train', validation_round) 


    inc_rate = np.array(weather_data_df['incubation'], dtype=np.float64)
    bite_rate = np.array(weather_data_df['bite'], dtype=np.float64)
    egg_laying_rate = np.array(weather_data_df['egg'], dtype=np.float64)
    egg_development_rate = np.array(weather_data_df['theta'], dtype=np.float64)

    if fit_weeks is not None:
        observed_cases = csv_state_cases_df['casos'].values[fit_weeks]
    else:
        observed_cases = csv_state_cases_df['casos'].values

    sim_data.egg_laying_rate = egg_laying_rate
    sim_data.egg_development_rate = egg_development_rate
    sim_data.bite_rate = bite_rate
    sim_data.inc_rate = inc_rate
    sim_data.initial_cases_df = initial_cases_df
    sim_data.initial_vectors_df = initial_vectors_df
    sim_data.days = days
    sim_data.csv_state_cases_df = csv_state_cases_df
    sim_data.fit_weeks = fit_weeks

    # Select prior configuration
    if prior_config not in prior_specs:
        print(f"Warning: Unknown prior_config '{prior_config}', using 'relaxed'")
        prior_config = "relaxed"

    priors = prior_specs[prior_config]

    if plot_priors:
        plot_prior_distributions_ready(priors)

    print(f"\n{'='*60}")
    print(f"Using '{prior_config}' prior configuration:")
    print(f"  k_v ~ Beta({priors['k_v'][0]}, {priors['k_v'][1]})")
    print(f"  k_h ~ Beta({priors['k_h'][0]}, {priors['k_h'][1]})")
    print(f"  s_0 ~ Beta({priors['s_0'][0]}, {priors['s_0'][1]})")
    print(f"  b_factor ~ Uniform({priors['b_factor'][0]}, {priors['b_factor'][1]})")
    print(f"  inc_factor ~ Uniform({priors['inc_factor'][0]}, {priors['inc_factor'][1]})")
    print(f"{'='*60}\n")


    # Load pre-computed MEM thresholds for the state
    thresholds = load_mem_thresholds_for_state(state)
    threshold_n = thresholds['epidemic_threshold']

    # Create a mask: True for weeks where cases > nth percentile
    mask = observed_cases > threshold_n

    print(f"Using {np.sum(mask)} out of {len(mask)} weeks for likelihood computation")
    print(f"Threshold: {threshold_n:.2f}")

    with pm.Model() as model:
        # Create priors using selected configuration
        k_v = pm.Beta("k_v", alpha=priors["k_v"][0], beta=priors["k_v"][1])
        k_h = pm.Beta("k_h", alpha=priors["k_h"][0], beta=priors["k_h"][1])
        s_0 = pm.Beta("s_0", alpha=priors["s_0"][0], beta=priors["s_0"][1])
        b_factor = pm.Uniform("b_factor", lower=priors["b_factor"][0], upper=priors["b_factor"][1])
        inc_factor = pm.Uniform("inc_factor", lower=priors["inc_factor"][0], upper=priors["inc_factor"][1])


        cases_mean = dengue_sim_fast(k_v, k_h, s_0, b_factor, inc_factor)
        cases_safe = pm.math.maximum(cases_mean, 1e-5)  # Slightly higher minimum
        cases_clipped = pm.math.clip(cases_safe, 1e-6, 1e7)

        # Apply mask to observed data
        observed_masked = observed_cases[mask]
        cases_clipped_masked = cases_clipped[mask]

        # Likelihood only on masked data
        Y_obs = pm.NegativeBinomial("Y_obs", mu=cases_clipped_masked, alpha=alpha_fixed, observed=observed_masked)

        prior_checks = pm.sample_prior_predictive(samples=1000, 
                                                  var_names=["k_v", "k_h", "s_0", 
                                                           "b_factor", "inc_factor"])

        # SMC runs one process per chain. We cap BOTH chains and cores to
        # N_CORES so we never spawn more workers than we allow (each worker
        # forks a full copy of the model + data, so chains drive RAM use).
        # draws reduced 2000 -> 1000: with N_CORES chains that is still
        # N_CORES*1000 posterior samples, plenty for these intervals, at
        # roughly half the peak memory and wall-clock of the original.
        trace = pm.sample_smc(
            draws=1000,
            chains=N_CORES,
            cores=N_CORES,
            progressbar=progress_bar_bool,
            random_seed=42,
            compute_convergence_checks=True,
        )

        idata = pm.to_inference_data(trace, model=model)
        idata = pm.compute_log_likelihood(idata, model=model, extend_inferencedata=True, progressbar=progress_bar_bool)

    # Log-likelihood analysis
    try:
        if hasattr(idata, "log_likelihood") and "Y_obs" in idata.log_likelihood:
            ll = idata.log_likelihood["Y_obs"].values
            chains, draws, obs = ll.shape
            ll_flat = ll.reshape(chains * draws, obs)
            total_ll_per_sample = ll_flat.sum(axis=1)

            waic = az.waic(idata)

            try:
                loo = az.loo(idata)

                log_likelihood_params = {'median': np.median(total_ll_per_sample),
                                 'std': np.std(total_ll_per_sample),
                                 'min': np.min(total_ll_per_sample),
                                 'max': np.max(total_ll_per_sample),
                                 'waic': waic.elpd_waic,
                                 'waic_se': waic.se,
                                 'loo': loo.elpd_loo,
                                 'loo_se': loo.se}
            except Exception as e:
                traceback.print_exception(e)
                print(f"Could not compute LOO: {e}")         
    except Exception as e:
        print(f"Error extracting log-likelihood: {e}")
        traceback.print_exc()

    print(pm.summary(trace, var_names=['k_v', 'k_h', 's_0', 'b_factor', 'inc_factor'], 
                    hdi_prob=0.95))

    # Generate ensemble fits.
    # Flatten each posterior array ONCE (the original flattened all five
    # arrays on every iteration, i.e. 5 * n_total redundant copies).
    kv_s   = trace.posterior.k_v.values.flatten()
    kh_s   = trace.posterior.k_h.values.flatten()
    s0_s   = trace.posterior.s_0.values.flatten()
    bf_s   = trace.posterior.b_factor.values.flatten()
    inc_s  = trace.posterior.inc_factor.values.flatten()
    n_total_samples = len(kv_s)

    bite_vals = weather_data_df['bite'].values
    inc_vals  = weather_data_df['incubation'].values

    fit_results = []
    bite_rate_adjusted = []
    incubation_rate_adjusted = []

    for idx in range(n_total_samples):
        sample_params = {
            'k_v':        kv_s[idx],
            'k_h':        kh_s[idx],
            's_0':        s0_s[idx],
            'b_factor':   bf_s[idx],
            'inc_factor': inc_s[idx],
        }

        weekly_fit, Nv_fit, Nh_fit = simulate_dengue_wrapper(
            sample_params, egg_laying_rate, egg_development_rate,
            bite_rate, inc_rate, initial_cases_df, initial_vectors_df, days
        )

        fit_results.append(weekly_fit)
        bite_rate_adjusted.append(bf_s[idx] * bite_vals)
        incubation_rate_adjusted.append(inc_s[idx] * inc_vals)

    fit_results = np.array(fit_results)

    # Calculate fit quality metrics (R², RMSE, MAE)
    observed_full = csv_state_cases_df['casos'].values
    predicted_median = np.median(fit_results, axis=0)

    min_len = min(len(observed_full), len(predicted_median))
    obs = observed_full[:min_len]
    pred = predicted_median[:min_len]

    # Calculate metrics
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))

    # R² (coefficient of determination)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Correlation coefficient
    correlation = np.corrcoef(pred, obs)[0, 1]

    # Create fit quality metrics dictionary
    fit_quality_metrics = {
        'R²': r_squared,
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Total_Observed': np.sum(obs),
        'Total_Predicted': np.sum(pred),
        'Prediction_Ratio': np.sum(pred) / np.sum(obs) if np.sum(obs) > 0 else np.nan
    }

    # Print fit quality metrics
    print(f"\n{'='*60}")
    print(f"MODEL FIT QUALITY METRICS - {state}")
    print(f"{'='*60}")
    print(f"  R² (Coefficient of Determination): {r_squared:.4f}")
    print(f"  Correlation:                        {correlation:.4f}")
    print(f"  RMSE (Root Mean Square Error):      {rmse:.2f}")
    print(f"  MAE (Mean Absolute Error):          {mae:.2f}")
    print(f"  Total Observed Cases:               {np.sum(obs):.0f}")
    print(f"  Total Predicted Cases:              {np.sum(pred):.0f}")
    print(f"  Prediction Ratio (Pred/Obs):        {fit_quality_metrics['Prediction_Ratio']:.4f}")
    print(f"{'='*60}\n")

    weather_data_df[['rhum', 'temp']].plot(figsize=(10, 4))
    plt.title(f"Humidity and Temperature for {state}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # Generate plots if requested
    if plot_results:        
        plot_posterior_analysis(trace, fit_results, csv_state_cases_df, state, 
                                weather_data_df, bite_rate_adjusted, 
                                incubation_rate_adjusted,
                                fit_weeks_indices=fit_weeks)

    # Combine all metrics to return
    combined_metrics = {**log_likelihood_params, **fit_quality_metrics}

    # Return full posterior samples
    posterior_samples = {
        'k_v':        trace.posterior.k_v.values.flatten(),
        'k_h':        trace.posterior.k_h.values.flatten(),
        's_0':        trace.posterior.s_0.values.flatten(),
        'b_factor':   trace.posterior.b_factor.values.flatten(),
        'inc_factor': trace.posterior.inc_factor.values.flatten(),
    }

    # Save complete model results for future use
    save_posterior_results(
        trace, csv_state_cases_df, weather_data_df, state, start_date, end_date,
        posterior_samples, timestamp
    )

    return timestamp, posterior_samples, combined_metrics


# Results storage
fit_results_by_state = {}      # Training results for each state
forecast_results_by_state = {} # Forecast results for each state


for state in brazilian_states:
    print(state)
    # === MODEL TRAINING PHASE ===
    state = adm_1_map[state]
    print(state)
    # prepare dict of results
    fit_results_by_state[state] = {}

    try:
        print(f"=== Fitting model for {state} ===")
        # Perform Bayesian parameter estimation
        timestamp, posterior_data, R0 = fit_function(state, train_start_date, train_end_date, geo_data)

        #Load fitted model components
        posterior_data, metadata, _ = load_posterior_results(timestamp, state)

        # Store training results
        fit_results_by_state[state] = {
            'timestamp': timestamp,
            'posterior_data': posterior_data,
            'metadata': metadata
        }
    except Exception as e:
        traceback.print_exception(e)
        print(f"Error fitting state {state}: {e}")

    # === FORECASTING PHASE ===
    try:
        print(f"=== Forecasting model for {state} ===")
        # Extract fitted model components
        posterior_data = fit_results_by_state[state]['posterior_data']
        metadata = fit_results_by_state[state]['metadata']

        # Generate probabilistic forecasts
        forecast_df, control_cases = forecast_function(state, forecast_start_date, forecast_end_date,
                                                      metadata, posterior_data, train_start_date, train_end_date, geo_data)

        # Store forecast results
        forecast_results_by_state[state] = {
            'forecast': forecast_df,
            'observed': control_cases
        }
    except Exception as e:
        traceback.print_exception(e)
        print(f"Error forecasting state {state}: {e}")    



    # prepare submission
    api_key = 'mattiamazzoli:2b9c6c97-bf22-4117-86f9-94e1353e70b9'
    repository = "mattiamazzoli/3rd_imdc_isi_isi-dengue"
    description = f"forecast_round_{validation_round}"
    commit = "04dbcdacbe95a0dbe19e2f1859a97ea16e49dfcb"
    adm_level = 1
    adm_0 = "BRA"
    adm_1 = int(state_code[state])

    # select only target weeks
    df_pred = forecast_df[(forecast_df.date >= forecast_start_date) & (forecast_df.date <= forecast_end_date)]
    df_pred = df_pred.drop(columns=['week','std'])
    df_pred.head()

    df_pred['state'] = state

    df_pred.to_csv(f'./model_outputs/round{validation_round}/{state}.csv', index=None)


# ── Cleanup: release memory and any worker resources after a run ──
# SMC's process pool is closed by PyMC automatically, but the large trace
# objects, matplotlib figures and InferenceData can pin hundreds of MB.
# Call this between states (or at the end) to keep peak RAM down.
import gc

def free_run_memory(*objs):
    """Drop references, close all figures, and force a GC pass."""
    plt.close('all')
    for o in objs:
        del o
    gc.collect()

# Example: free_run_memory()  # after you no longer need the last trace/idata
free_run_memory()



