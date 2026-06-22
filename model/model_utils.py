#Dengue Transmission Model (Compartmental ODE System)
#===================================================
#from sympy import exp  # pyright: ignore[reportMissingImports]
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import pandas as pd
import time
from numba import njit
import warnings
warnings.filterwarnings('ignore')
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import scipy.stats as st
import os, pickle, json
#import mosqlient as mosq
from datetime import datetime
#from geopy import Nominatim
#from meteostat import Stations, Hourly
import arviz as az  
import corner  
import traceback
from scipy.stats import norm  # pyright: ignore[reportMissingImports]
from scipy.stats import beta, uniform # pyright: ignore[reportMissingImports]
import traceback
from pathlib import Path
import string
from typing import Any
import xarray as xr 

prior_specs = {
    "very_relaxed": {
        "k_v": (1, 1),           # Beta(1,1) = Uniform(0,1)
        "k_h": (1, 1),           # Beta(1,1) = Uniform(0,1)
        "s_0": (1, 1),       # Beta(1,1) = Uniform(0,1)
        "b_factor": (0, 0.1),    # Much wider range
        "inc_factor": (0, 0.001) # Much wider range
    },
    "relaxed": {
        "k_v": (2, 8),           # Mean ~0.2, wider spread
        "k_h": (3, 3),           # Mean ~0.5, symmetric
        "s_0": (2, 3),       # Mean ~0.4
        "b_factor": (0, 0.05),   # 5x wider
        "inc_factor": (1e-5, 5e-4)  # ~5x wider
    },
    "moderate": {
        "k_v": (3, 12),          # Mean ~0.2
        "k_h": (5, 4),           # Mean ~0.55
        "s_0": (2, 4),       # Mean ~0.33
        "b_factor": (0, 0.02),   # 2x wider
        "inc_factor": (2e-5, 2e-4)  # ~2x wider
    },
    "informative": {
        "k_v": (2, 20),          # Original
        "k_h": (8, 5.5),         # Original
        "s_0": (2, 5),        # Original (2, 5),
        "b_factor": (0, 1e-2),   # Original  ,
        #"inc_factor": (3.6e-5, 1.09e-4)  # Original
        "inc_factor": (1.5e-5, 5.5e-5)  # Original
        
    }
}
            
@njit(fastmath=True, cache=True)
def ode_system(y, k_v, k_h, pi_v, theta_v, b, inc):
    """
    Dengue transmission ODE system with vector and human compartments.
    
    Vector compartments:
    - Pv: Pre-emergence (pupae/larvae)
    - Sv: Susceptible vectors  
    - Ev: Exposed vectors (infected but not infectious)
    - Qv: Pre-emergence from infected vectors (vertical transmission)
    - Iv: Infectious vectors
    - Dv: Dead vectors (cumulative)
    
    Human compartments:
    - Sh: Susceptible humans
    - Eh: Exposed humans (infected but not infectious)
    - Ah: Undetected humans (infected, no symptoms, can transmit)
    - Ih: Infectious/detected humans
    - Rh: Recovered humans
    - Dh: Dead humans (cumulative)
    
    Parameters:
    -----------
    y : array
        Current state vector [Pv, Sv, Ev, Qv, Iv, Dv, Sh, Eh, Ah, Ih, Rh, Dh]
    """
    Pv, Sv, Ev, Qv, Iv, Dv, Sh, Eh, Ah, Ih, Rh, Dh = y
    
    alpha_v = inc  # Vector incubation rate (temperature-dependent)
    
    # Calculate population sizes
    Nv = Sv + Ev + Iv  # Total vector population
    Nh = Sh + Eh + Ah + Ih + Rh  # Total human population
    
    # Density-dependent mortality (competition for resources)
    mortality_factor = mu_v * (Nv / (Nh * cc_v))
    
    # Force of infection (transmission rates)
    # Both detected and undetected can transmit to vectors
    infection_rate_v = b * k_v * ((Ih + Ah) / Nh)  # Human to vector
    infection_rate_h = b * k_h * (Iv / Nh)  # Vector to human
    
    # Proportion of underreporting (50%)
    under_reporting_rate = 0.5
    
    # Vector compartment dynamics
    dpv = egg_lper * pi_v * (Nv - psi_v * Iv) - (mu_e + female_per * theta_v) * Pv
    dsv = female_per * theta_v * Pv - (mortality_factor + infection_rate_v) * Sv
    dev = infection_rate_v * Sv - (mortality_factor + alpha_v) * Ev
    dqv = egg_lper * pi_v * psi_v * Iv - (mu_e + female_per * theta_v) * Qv
    div = female_per * theta_v * Qv + alpha_v * Ev - mortality_factor * Iv
    ddv = mortality_factor * Iv
    
    # Human compartment dynamics
    dsh = pi_h * Nh - (mu_h + infection_rate_h) * Sh
    deh = infection_rate_h * Sh - (mu_h + alpha_h) * Eh
    
    # 50% of exposed go undetected, 50% become detected
    dah = under_reporting_rate * alpha_h * Eh - (mu_h + beta_h + sigma_h) * Ah
    dih = (1 - under_reporting_rate) * alpha_h * Eh - (mu_h + beta_h + sigma_h) * Ih
    
    # Recovery from both undetected and detected
    drh = beta_h * (Ah + Ih) - mu_h * Rh
    
    # Deaths only from detected cases
    ddh = sigma_h * Ih
    
    return np.array([dpv, dsv, dev, dqv, div, ddv, dsh, deh, dah, dih, drh, ddh])

@njit(fastmath=True, cache=True)
def rk4_step(y, T, dt, k_v, k_h, Pi_v, Theta_v, Bite_rate, Inc_rate):
    """
    Fourth-order Runge-Kutta integration step.
    """
    
    pi_v = Pi_v[T]
    theta_v = Theta_v[T]
    b = Bite_rate[T]
    inc = Inc_rate[T]

    k1 = ode_system(y, k_v, k_h, pi_v, theta_v, b, inc)
    k2 = ode_system(y + 0.5 * dt * k1, k_v, k_h, pi_v, theta_v, b, inc)
    k3 = ode_system(y + 0.5 * dt * k2, k_v, k_h, pi_v, theta_v, b, inc)
    k4 = ode_system(y + dt * k3, k_v, k_h, pi_v, theta_v, b, inc)
    
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@njit(fastmath=True, cache=True)
def initial_conditions_fast(tot_cases, tot_pop, tot_vectors, tot_eggs, s_0):
    """
    Calculate epidemiologically consistent initial conditions.
    
    Assumes 50% of infected individuals are undetected.
    Observed cases represent only detected infections.
    
    Parameters:
    -----------
    tot_cases : float
        Total observed cases in first week (detected only)
    tot_pop : float
        Total population size
    tot_vectors : float
        Total vector population
    tot_eggs : float
        Total egg population
    s_0 : float
        Proportion of susceptible population
        
    Returns:
    --------
    y0 : array
        Initial state vector: [Pv, Sv, Ev, Qv, Iv, Dv, Sh, Eh, Ah, Ih, Rh, Dh]
    """
    Nh = tot_pop                       # Total human population
    Nv = tot_vectors                   # Total vector population
    Ne = tot_eggs                      # Total egg population

    # Estimate initial detected humans from case data
    ih0 = max(int(tot_cases), 1)       # Infectious detected humans
    
    # Assume equal numbers of undetected (since 50% split)
    ah0 = ih0                          # undetected humans
    
    # Total exposed is sum of those who will become detected and undetected
    eh0 = int((ih0 + ah0) / alpha_h)   # Exposed humans
    
    sh0 = int(s_0 * Nh)            # Susceptible humans
    rh0 = int(Nh - (ih0 + ah0 + eh0 + sh0))  # Recovered humans

    # Calculate adult vector populations proportional to total human infections
    total_infectious_humans = ih0 + ah0
    iv0 = int(cc_v * total_infectious_humans)  # Infectious vectors
    ev0 = int(cc_v * eh0)                      # Exposed vectors
    sv0 = int(Nv - (iv0 + ev0))                # Susceptible vectors

    # Calculate aquatic vector populations
    qv0 = int(cc_v * (ih0 + ah0 + eh0))  # Infected eggs (vertical transmission)
    pv0 = int(Ne - qv0)                   # Normal eggs

    return np.array([float(pv0), float(sv0), float(ev0), float(qv0), float(iv0), 0.0, 
                     float(sh0), float(eh0), float(ah0), float(ih0), float(rh0), 0.0])
    
@njit(fastmath=True, cache=True)
def simulate_dengue_fast(k_v, k_h, s_0, b_factor, inc_factor,
                         tot_cases, tot_pop, tot_vectors, tot_eggs,
                         egg_lrate, egg_drate, bite_rate, inc_rate,
                         days, dt = 1.0):
    """
    Fast simulation of dengue transmission dynamics with undetected compartment.
    
    NOW RETURNS 12 COMPARTMENTS: [Pv, Sv, Ev, Qv, Iv, Dv, Sh, Eh, Ah, Ih, Rh, Dh]
    
    Parameters:
    -----------
    k_v, k_h, s_0 : float
        Model parameters to be estimated
    b_factor, inc_factor : float
        Adjustment factors for weather parameters
    tot_cases, tot_pop : float
        Initial epidemic conditions
    tot_vectors, tot_eggs : float
        Initial vector populations
    egg_lrate, egg_drate, bite_rate, inc_rate : array
        Time-varying weather-dependent parameters
    days : int
        Simulation duration
    dt : float
        Time step (default 1.0 day)
        
    Returns:
    --------
    results : array
        Time series of all 12 compartments [days x 12]
    dt : float
        Time step used
    """
    # Initialize with epidemiologically consistent conditions
    y = initial_conditions_fast(tot_cases, tot_pop, tot_vectors, tot_eggs, s_0)
    time_steps = int(days / dt)
    
    # Updated to 12 compartments
    results = np.zeros((time_steps + 1, 12))
    results[0] = y
    
    # Integrate ODE system day by day
    for i in range(1, time_steps + 1):
        # Use weather parameters for current day (with bounds checking)
        idx = min(i, len(egg_lrate) - 1)
        pi_v = egg_lrate
        theta_v = egg_drate
        b = bite_rate * b_factor
        inc = inc_rate * inc_factor
        
        # Runge-Kutta integration step
        y = rk4_step(y, idx, dt, k_v, k_h, pi_v, theta_v, b, inc)
        # Prevent negative populations
        y = np.maximum(y, MIN_VALUE)
        
        results[i] = y

    return results, dt

@njit(fastmath=True, cache=True)
def calculate_weekly_cases(results, dt):
    """
    Calculate weekly case incidence from compartment dynamics.
    NOW ACCOUNTS FOR UNDETECTED COMPARTMENT (Ah at index 8)
    
    IMPORTANT: Returns only detected cases to match surveillance data!
    Undetected cases are not reported in surveillance systems.
    
    Parameters:
    -----------
    results : array
        Time series from ODE simulation (12 compartments)
    dt : float
        Time step used in simulation
        
    Returns:
    --------
    weeks : array
        Week numbers
    weekly_cases : array
        Weekly detected case counts (for comparison with observed data)
    """
    # Calculate number of steps per day and per week
    steps_per_day = int(1.0 / dt)
    steps_per_week = 7 * steps_per_day
    
    # Extract required compartments (UPDATED INDICES)
    Ah = results[:, 8]  # Undetected humans
    Ih = results[:, 9]  # Infectious detected humans
    Rh = results[:, 10] # Recovered humans  
    Dh = results[:, 11] # Dead humans
    
    total_steps = len(results)
    missing_days = total_steps % steps_per_week
    total_weeks = total_steps // steps_per_week

    # Initialize arrays for daily and weekly cases
    daily_cases = np.zeros(total_steps)
    weekly_cases = np.zeros(total_weeks + (1 if missing_days > 0 else 0))
    
    # Calculate daily incidence (detected ONLY - matches surveillance data)
    # Undetected cases are NOT counted as they're not detected
    daily_cases[0] = Ih[0]
    for i in range(1, total_steps):
        # Only count detected cases for comparison with observed data
        new_Ih = max(0.0, Ih[i] - Ih[i-1])
        
        # Approximate recovery flow from detected (50% of total recovery)
        new_Rh_from_Ih = max(0.0, Rh[i] - Rh[i-1]) * 0.5
        
        # Deaths only from detected
        new_Dh = max(0.0, Dh[i] - Dh[i-1])
        
        daily_cases[i] = new_Ih + new_Rh_from_Ih + new_Dh

    # Aggregate cases weekly accounting for dt
    weekly_cases[0] = np.sum(daily_cases[0:missing_days]) if missing_days > 0 else daily_cases[0]
    for week in range(total_weeks):
        start_idx = week * steps_per_week + missing_days
        end_idx = start_idx + steps_per_week
    
        weekly_cases[week+1] = np.sum(daily_cases[start_idx:end_idx])
        
    weeks = np.arange(total_weeks + (1 if missing_days > 0 else 0))
    
    return weeks, weekly_cases


@njit(fastmath=True, cache=True)
def Basic_Reproduction_Number_fast(k_h, k_v, mean_bite, Nv, Nh):
    """
    Calculate basic reproduction number (R0) for dengue transmission.
    
    NOTE: With undetected compartment, both Ah and Ih contribute to transmission.
    This R0 calculation assumes the combined infectiousness of detected + undetected.
    
    R0 represents the expected number of secondary infections produced by
    one infected individual in a completely susceptible population.
    
    For vector-borne diseases, R0 includes both human-vector and vector-human
    transmission cycles.
    
    Parameters:
    -----------
    k_h, k_v : float
        Transmission probabilities
    mean_bite : float
        Average biting rate
    Nv, Nh : float
        Vector and human population sizes
        
    Returns:
    --------
    R0 : float
        Basic reproduction number
    """
    if Nh < 1:
        Nh = 1
    if Nv < 1:
        Nv = 1
    
    # Vector infection probability and duration
    # k_v already accounts for combined transmission from Ih + Ah
    term1 = (mean_bite * k_v * Nv) / (Nh * (alpha_v + mu_v * Nv / (Nh * cc_v)))
    
    # Human infection probability
    term2 = (mean_bite * k_h) / (alpha_h + mu_h)
    
    # Vector infectious duration
    term3 = alpha_v / (mu_v * Nv / (Nh * cc_v))
    
    # Human infectious duration (average of detected and undetected)
    # undetected: 1/(mu_h + beta_h)
    # detected: 1/(mu_h + beta_h + sigma_h)
    # Average with 50% split
    term4_asymp = alpha_h / (mu_h + beta_h)
    term4_symp = alpha_h / (beta_h + mu_h + sigma_h)
    term4 = 0.5 * (term4_asymp + term4_symp)
    
    # R0 is geometric mean of transmission cycle components
    R0 = np.sqrt(term1 * term2 * term3 * term4)
    
    return R0


#Model Training and Parameter Estimation
#=======================================
def plot_prior_distributions_ready(prior_dict, n_samples=10000):
    """
    Plot prior distributions for Bayesian dengue model parameters
    in a 2x3 grid (5 priors + 1 empty subplot).

    Parameters
    ----------
    prior_dict : dict
        Dictionary of priors (e.g. prior_specs["informative"])
    n_samples : int
        Number of samples drawn from each prior
    """
    

    # --- Draw samples ---
    samples = {}
    for key, val in prior_dict.items():
        if key in ["k_v", "k_h", "s_0"]:
            a, b = val
            samples[key] = beta.rvs(a, b, size=n_samples)
        elif key in ["b_factor", "inc_factor"]:
            low, high = val
            samples[key] = uniform.rvs(loc=low, scale=high - low, size=n_samples)

    # --- Create 2x3 layout ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    # --- Plot histograms ---
    for i, key in enumerate(prior_dict.keys()):
        ax = axes[i]
        ax.hist(samples[key],
                bins=50,
                density=True,
                alpha=0.7,
                edgecolor="black")

        if key in ["k_v", "k_h", "s_0"]:
            ax.set_title(rf"${key} \sim \mathrm{{Beta}}{prior_dict[key]}$")
        elif key == "inc_factor":
            low, high = prior_dict[key]
            low_s = f"{low:.2e}".split("e")
            high_s = f"{high:.2e}".split("e")

            ax.set_title(
                rf"$inc_{{factor}} \sim \mathrm{{Uniform}}({low_s[0]}\times10^{{{int(low_s[1])}}}, {high_s[0]}\times10^{{{int(high_s[1])}}})$"
            )
        else:
            main, sub = key.split("_", 1)
            ax.set_title(rf"${main}_{{{sub}}} \sim \mathrm{{Uniform}}{prior_dict[key]}$")

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    # --- Turn off last empty subplot ---
    axes[-1].axis("off")

    plt.suptitle(
        r"Prior Distributions of Model Parameters",
        fontweight="bold"
    )
    plt.show()

def plot_prior_paper_ready(prior_dict, n_samples=10000):
    samples = {}
    for key, val in prior_dict.items():
        if key in ["k_v", "k_h", "s_0"]:
            a, b = val
            samples[key] = beta.rvs(a, b, size=n_samples)
        elif key in ["b_factor", "inc_factor"]:
            low, high = val
            samples[key] = uniform.rvs(loc=low, scale=high-low, size=n_samples)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    panel_labels = list(string.ascii_uppercase)

    for i, key in enumerate(prior_dict.keys()):

        ax = axes[i]

        ax.hist(
            samples[key],
            bins=50,
            density=True,
            alpha=0.75,
            edgecolor="black"
        )

        # Panel label (A, B, C...)
        ax.text(
            0.02, 0.95,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top"
        )

        if key in ["k_v", "k_h", "s_0"]:
            ax.set_title(rf"${key} \sim \mathrm{{Beta}}{prior_dict[key]}$", fontsize=12)

        elif key == "inc_factor":
            low, high = prior_dict[key]

            low_s = f"{low:.2e}".split("e")
            high_s = f"{high:.2e}".split("e")

            ax.set_title(
                rf"$inc_{{factor}} \sim \mathrm{{Uniform}}({low_s[0]}\times10^{{{int(low_s[1])}}}, {high_s[0]}\times10^{{{int(high_s[1])}}})$",
                fontsize=12
            )

            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        else:
            main, sub = key.split("_",1)
            ax.set_title(
                rf"${main}_{{{sub}}} \sim \mathrm{{Uniform}}{prior_dict[key]}$",
                fontsize=12
            )

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    axes[-1].axis("off")

    fig.suptitle(
        "Prior Distributions of Model Parameters",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
 
def plot_posterior_analysis(trace, fit_results, csv_state_cases_df, state, 
                            weather_data_df, bite_rate_adjusted, incubation_rate_adjusted,
                            fit_weeks_indices=None):
    """
    Plot comprehensive posterior analysis including:
    - Trace plots
    - Fit vs observed cases with confidence intervals
    - Adjusted weather parameters
    - Corner plot for parameter correlations
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        MCMC trace from PyMC sampling
    fit_results : np.array
        Array of fit trajectories from posterior samples
    csv_state_cases_df : pd.DataFrame
        Observed case data
    state : str
        State name for plot titles
    weather_data_df : pd.DataFrame
        Weather data with temporal index
    bite_rate_adjusted : list
        Adjusted biting rates for each posterior sample
    incubation_rate_adjusted : list
        Adjusted incubation rates for each posterior sample
    """
    
    # 1. Trace plots for convergence diagnostics
    az.plot_trace(trace, var_names=['k_v', 'k_h', 's_0', 'b_factor', 'inc_factor'])
    plt.tight_layout()
    plt.show()

    # 2. Fit vs observed cases with confidence intervals
    confidence_levels = [50, 80, 90, 95]
    fit_stats = {}
    for level in confidence_levels:
        lower_bound = 50 - level / 2
        upper_bound = 50 + level / 2
        fit_stats[f'ci_{level}_lower'] = np.percentile(fit_results, lower_bound, axis=0)
        fit_stats[f'ci_{level}_upper'] = np.percentile(fit_results, upper_bound, axis=0)

    fit_stats['median'] = np.median(fit_results, axis=0)
    fit_stats['std'] = np.std(fit_results, axis=0)

    fit_df = pd.DataFrame({
        'week': range(len(fit_stats['median'])),
        'median': fit_stats['median'],
        'std': fit_stats['std']
    })
    
    for level in confidence_levels:
        fit_df[f'ci_{level}_lower'] = fit_stats[f'ci_{level}_lower']
        fit_df[f'ci_{level}_upper'] = fit_stats[f'ci_{level}_upper']

    fig, ax = plt.subplots(figsize=(12, 7))
    
    weeks = np.arange(len(csv_state_cases_df['casos']))
    dates = pd.to_datetime(csv_state_cases_df['data_iniSE'])
    
    if fit_weeks_indices is not None:
        fit_idx = np.array(fit_weeks_indices)
        mask_fit = np.isin(weeks, fit_idx)
    else:
        mask_fit = np.ones(len(weeks), dtype=bool)
    mask_not_fit = ~mask_fit

    ax.fill_between(dates, fit_df['ci_95_lower'], fit_df['ci_95_upper'], color='#9ecae1',
                    alpha=0.35, label='95% PI', zorder=1)
    ax.plot(dates, fit_df['median'], color='#3182bd', linewidth=2.5, label='Model', zorder=2)
    ax.scatter(dates[mask_fit], csv_state_cases_df['casos'][mask_fit], color='black', s=40,
               label='Data (fit)', zorder=3)
    ax.scatter(dates[mask_not_fit], csv_state_cases_df['casos'][mask_not_fit], facecolors='white',
               edgecolors='black', s=40, label='Data (not fit)', zorder=3)
    ax.set_xlabel('Week')
    ax.set_ylabel('Detected Cases')
    ax.set_title(f'Model Fit vs Observed Cases - {state}')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    #plt.xticks(rotation=45)
    data_legend = Line2D(
    [0], [0],
    marker='o',
    linestyle='None',
    markerfacecolor='black',
    markeredgecolor='black',
    markersize=7,
    label='Data'
    )
    data_legend_white = Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markerfacecolor='white',
        markeredgecolor='black',
        markersize=7
    )
    legend_elements = [
        (data_legend, data_legend_white),
        Line2D([0], [0], color='#3182bd', lw=2.5, label='Model'),
        Line2D([0], [0], color='#9ecae1', lw=10, alpha=0.35, label='95% PI')
    ]
    ax.legend(
        legend_elements,
        ['Data', 'Model', '95% PI'],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False
    )
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    # 3. Adjusted weather parameters
    _, (bx1, bx2) = plt.subplots(1, 2, figsize=(16, 6))

    bx1.plot(weather_data_df.index, np.median(bite_rate_adjusted, axis=0), 
            label='Adjusted Biting Rate', color='orange', linewidth=2)
    bx1.set_xlabel('Date')
    bx1.set_ylabel('Biting Rate')
    bx1.set_title(f'Adjusted Biting Rate - {state}')
    bx1.legend()
    bx1.grid(True, alpha=0.3)
    
    bx2.plot(weather_data_df.index, np.median(incubation_rate_adjusted, axis=0), 
            label='Adjusted Incubation Rate', color='green', linewidth=2)
    bx2.set_xlabel('Date')
    bx2.set_ylabel('Incubation Rate')
    bx2.set_title(f'Adjusted Incubation Rate - {state}')
    bx2.legend()
    bx2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 4. Corner plot for parameter correlations
    k_v_samples = trace.posterior.k_v.values.flatten()
    k_h_samples = trace.posterior.k_h.values.flatten()
    s0_samples = trace.posterior.s_0.values.flatten()
    b_factor_samples = trace.posterior.b_factor.values.flatten()
    inc_factor_samples = trace.posterior.inc_factor.values.flatten()

    samples = np.column_stack([k_v_samples, k_h_samples, s0_samples, 
                            b_factor_samples, inc_factor_samples])
    labels = [r'$k_v$', r'$k_h$', r'$s_0$', r'$b_{factor}$', r'$inc_{factor}$']
    
    # Remove outliers (1-99 percentile)
    lower_bounds = np.percentile(samples, 1, axis=0)
    upper_bounds = np.percentile(samples, 99, axis=0)
    mask = np.all((samples >= lower_bounds) & (samples <= upper_bounds), axis=1)
    samples_clipped = samples[mask]

    if samples_clipped.shape[0] < 10:
        samples_clipped = samples

    range_per_param = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    
    fig = corner.corner(
        samples_clipped,
        labels=labels,
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        bins=40,
        smooth=1.0,
        range=range_per_param,
        color="green",
    )

    n = len(labels)
    axes = np.array(fig.axes).reshape((n, n))

    sci_indices = [n - 2, n - 1]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_useOffset(False)
    formatter.set_powerlimits((0, 0))
    formatter.set_powerlimits((-2, 2))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i < n - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")

            if j > 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")

            if i in sci_indices or j in sci_indices:
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)

    # fix overlap for the last parameter (inc_factor diagonal)
    ax_last = axes[n-1, n-1]
    ax_last.xaxis.get_offset_text().set_y(1.05)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.98,
        wspace=0.30,
        hspace=0.30,
    )

    plt.suptitle(f'Posterior Distributions - {state}', y=1.02)
    plt.show()

@njit(fastmath=True, cache=True)
def simulate_dengue_fast_from_scenario(k_v, k_h, s_0, b_factor, inc_factor,
                                       final_state_training,
                                       egg_lrate, egg_drate, bite_rate, inc_rate,
                                       days, dt=1.0):
    """
    Fast simulation of dengue transmission using final training state as initial condition.
    
    Parameters:
    -----------
    k_v, k_h, s_0 : float
        Model parameters from posterior
    b_factor, inc_factor : float
        Adjustment factors for weather parameters
    final_state_training : array
        Final state from training period [Pv, Sv, Ev, Qv, Iv, Dv, Sh, Eh, Ah, Ih, Rh, Dh]
    egg_lrate, egg_drate, bite_rate, inc_rate : array
        Time-varying weather-dependent parameters for scenario period
    days : int
        Simulation duration
    dt : float
        Time step (default 1.0 day)
        
    Returns:
    --------
    results : array
        Time series of all 12 compartments [days x 12]
    dt : float
        Time step used
    """
    # Initialize from final training state (ensures continuity)
    y = initial_conditions_fast_from_scenario(final_state_training)

    time_steps = int(days / dt)
    
    results = np.zeros((time_steps + 1, 12))
    results[0] = y
    
    # Integrate ODE system day by day
    for i in range(1, time_steps + 1):
        idx = min(i, len(egg_lrate) - 1)
        pi_v = egg_lrate
        theta_v = egg_drate
        b = bite_rate * b_factor
        inc = inc_rate * inc_factor
        
        y = rk4_step(y, idx, dt, k_v, k_h, pi_v, theta_v, b, inc)
        y = np.maximum(y, MIN_VALUE)
        
        results[i] = y

    return results, dt


# Vectors
#==================
def load_or_fetch_vectors(state, geo_data, start_date, end_date, mode):
    """
    Load cached cases data for a state if available,
    otherwise fetch and cache it.
    Uses 'time' as index.
    """

    vectors_state_dir = f"./data_imdc_2026/{state}"
    os.makedirs(vectors_state_dir, exist_ok=True)
    if mode == 'train':
        vectors_cache_file = os.path.join(vectors_state_dir, "vectors.csv")
    elif mode == 'forecast':
        vectors_cache_file = os.path.join(vectors_state_dir, "vectors_forecast.csv")
    
    if os.path.exists(vectors_cache_file):
        print(f"Using cached vectors data for {state}")
        df = pd.read_csv(vectors_cache_file, parse_dates=True)
    else:
        start_date_fetch = "2010-01-03"
        end_date_fetch = end_date

        previous_week_date = str(datetime.date(datetime.strptime(start_date_fetch, "%Y-%m-%d") - timedelta(days=1)))
        weather_start_date = str(datetime.date(datetime.strptime(previous_week_date, "%Y-%m-%d") - timedelta(days=2)))
        date_difference = datetime.strptime(end_date_fetch, "%Y-%m-%d") - datetime.strptime(start_date_fetch, "%Y-%m-%d")
        days = date_difference.days

        if days % 7 != 0:
            end_date_fetch = str(datetime.date(datetime.strptime(end_date_fetch, "%Y-%m-%d") - timedelta(days=days % 7) + timedelta(days=7)))
            date_difference = datetime.strptime(end_date_fetch, "%Y-%m-%d") - datetime.strptime(start_date_fetch, "%Y-%m-%d")
            days = date_difference.days

        geo_data_state = geo_data[geo_data['uf'] == state]
        state_geocodes = geo_data_state['geocode'].astype(int).tolist()

        cases_df, major_cities = load_or_fetch_cases(
            state,
            state_geocodes,
            pd.to_datetime(previous_week_date),
            pd.to_datetime(end_date)
        )
        observed_cases = cases_df['casos'].to_numpy()
        
        weather_data_df = load_or_fetch_weather(
            state,
            start_date_fetch,
            end_date, 
            dict_weather_coeffs, 
            major_cities,
            mode
        )
    
        inc_rate = np.array(weather_data_df['incubation'], dtype=np.float64)
        bite_rate = np.array(weather_data_df['bite'], dtype=np.float64)
        egg_laying_rate = np.array(weather_data_df['egg'], dtype=np.float64)
        egg_development_rate = np.array(weather_data_df['theta'], dtype=np.float64)

        results, dt = simulate_vectors_fast(
            tot_pop=cases_df['pop'][0],
            cc_v_param=cc_v,
            egg_lrate=egg_laying_rate,
            egg_drate=egg_development_rate,
            days=days,
            dt=1.0
        )
        
        df = pd.DataFrame(results, columns=['Pv', 'Sv', 'Sh'])
        df['data_iniSE'] = pd.date_range(start=start_date_fetch, periods=len(df), freq='D')
        # ordina colonne come prima
        df = df[['data_iniSE', 'Pv', 'Sv', 'Sh']]

        # Filter BEFORE saving to cache
        mask = (df['data_iniSE'] >= start_date) & (df['data_iniSE'] <= end_date)
        df = df.loc[mask].reset_index(drop=True)

        # Filter BEFORE saving to cache
        df.to_csv(vectors_cache_file, index=False)
        print(f"Saved vector data to {vectors_cache_file}")

    # If cached file exists, filter it
    mask = (df['data_iniSE'] >= start_date) & (df['data_iniSE'] <= end_date)
    df = df.loc[mask].reset_index(drop=True)
    
    return df


    
@njit(fastmath=True, cache=True)
def simulate_vectors_fast(tot_pop, cc_v_param,
                          egg_lrate, egg_drate,
                          days, dt=1.0):
    """
    Simulazione veloce delle dinamiche vettoriali (Pv, Sv, Sh) usando RK4.
    Restituisce matrice (steps+1 x 3) con colonne [Pv, Sv, Sh] e il dt usato.
    Parametri:
      - tot_pop: popolazione umana iniziale (float)
      - cc_v_param: capacità vettoriale (int/float, es. cc_v)
      - tot_vectors: vettori adulti iniziali (float)
      - tot_eggs: uova iniziali (float)
      - egg_lrate: array tempo-variabile tasso deposizione uova (Pi_v)
      - egg_drate: array tempo-variabile tasso sviluppo uova (Theta_v)
      - days: durata simulazione in giorni (int)
      - dt: passo temporale in giorni (float, default 1.0)
    """
    # inizializza condizioni con la funzione esistente
    y = vectors_initial_conditions(tot_pop, cc_v_param).astype(np.float64)
    
    time_steps = int(days / dt)
    
    results = np.zeros((time_steps + 1, 3), dtype=np.float64)
    results[0] = y

    for i in range(1, time_steps + 1):
        idx = min(i, len(egg_lrate) - 1)
        pi_v = egg_lrate
        theta_v = egg_drate
        
        y = vectors_rk4_step(y, idx, dt, pi_v, theta_v)
        y = np.maximum(y, MIN_VALUE)
        
        results[i] = y

    return results, dt

@njit(fastmath=True, cache=True)
def vectors_initial_conditions(pop, cc_v):
    Nh = float(pop)                         # Total human population
    Sh0 = Nh                                # Susceptible humans

    Nv = cc_v * Nh                          # Total vector population
    Sv0 = np.round(cc_v * (Sh0), 0)   # Susceptible vectors
    Pv0 = np.round(cc_v * (Sh0), 0)   # Normal eggs

    y0 = np.array([Pv0, Sv0, Sh0])

    return y0

@njit(fastmath=True, cache=True)
def vectors_ode(y, pi_v, theta_v):
    Pv, Sv, Sh = y
    Nv = Sv
    Nh = Sh
    
    mortality_factor = mu_v * (Nv / (Nh * cc_v))
    
    dPv = egg_lper * pi_v * Nv - (mu_e + female_per * theta_v) * Pv
    dSv = female_per * theta_v * Pv - mortality_factor * Sv

    dSh = pi_h * Nh - mu_h * Sh

    return np.array([dPv, dSv, dSh])

    
@njit(fastmath=True, cache=True)
def vectors_rk4_step(y, T, dt, Pi_v, Theta_v):
    pi_v = Pi_v[T]
    theta_v = Theta_v[T]

    k1 = vectors_ode(y, pi_v, theta_v)
    k2 = vectors_ode(y + 0.5 * dt * k1, pi_v, theta_v)
    k3 = vectors_ode(y + 0.5 * dt * k2, pi_v, theta_v)
    k4 = vectors_ode(y + dt * k3, pi_v, theta_v)
    
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def save_posterior_results(trace, state_cases_df, weather_data_df, state, start_date, end_date, 
                          best_fit_params, timestamp, save_dir="./saved_models/"):
    """
    Save Bayesian posterior samples and metadata for future forecasting.
    
    Preserves all information needed to generate forecasts without re-fitting
    the model, including parameter samples, training data, and model configuration.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        PyMC posterior samples from MCMC
    state_cases_df : pd.DataFrame
        Training case data
    weather_data_df : pd.DataFrame
        Training weather data
    state : str
        State/region name
    start_date, end_date : str
        Training period dates
    best_fit_params : dict
        Maximum a posteriori parameter estimates
    R0_best_fit : float
        R0 calculated with best-fit parameters
    save_dir : str
        Directory for saved model files
        
    Returns:
    --------
    timestamp : str
        Unique identifier for saved model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract posterior parameter samples for forecasting
    posterior_data = {
        'k_v_samples': trace.posterior.k_v.values.flatten(),
        'k_h_samples': trace.posterior.k_h.values.flatten(),
        's_0_samples': trace.posterior.s_0.values.flatten(),
        'b_factor_samples': trace.posterior.b_factor.values.flatten(),
        'inc_factor_samples': trace.posterior.inc_factor.values.flatten(),
        'n_samples': len(trace.posterior.k_v.values.flatten())
    }
    
    # Store model metadata and configuration
    metadata = {
        'state': state,
        'training_start_date': start_date,
        'training_end_date': end_date,
        'best_fit_params': best_fit_params,
        'training_data_shape': state_cases_df.shape,
        'weather_coeffs': dict_weather_coeffs,
    }
    
    # Store training data for reference
    training_data = {
        'state_cases_df': state_cases_df.to_dict(),
        'weather_data_df': weather_data_df.to_dict()
    }
    
    
    # Save files with pickle for complex objects, JSON for metadata
    with open(f"{save_dir}posterior_samples_{state}_{timestamp}.pkl", 'wb') as f:
        pickle.dump(posterior_data, f)
    
    with open(f"{save_dir}metadata_{state}_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    with open(f"{save_dir}training_data_{state}_{timestamp}.pkl", 'wb') as f:
        pickle.dump(training_data, f)
    
    return timestamp

def load_posterior_results(timestamp, state, save_dir="./saved_models/"):
    """
    Load previously saved model results for forecasting.
    
    Reconstructs all necessary components for generating forecasts from
    a previously trained model.
    
    Parameters:
    -----------
    timestamp : str
        Model identifier from training
    state : str
        State/region name
    save_dir : str
        Directory containing saved models
        
    Returns:
    --------
    posterior_data : dict
        Parameter samples from posterior distribution
    metadata : dict
        Model configuration and training information
    training_data : dict
        Original training datasets
    """
    with open(f"{save_dir}posterior_samples_{state}_{timestamp}.pkl", 'rb') as f:
        posterior_data = pickle.load(f)
    
    with open(f"{save_dir}metadata_{state}_{timestamp}.json", 'r') as f:
        metadata = json.load(f)
    
    with open(f"{save_dir}training_data_{state}_{timestamp}.pkl", 'rb') as f:
        training_data = pickle.load(f)
    
    return posterior_data, metadata, training_data

    