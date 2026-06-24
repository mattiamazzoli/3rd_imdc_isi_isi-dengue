# Model Parameters
#=====================================
import pandas as pd
import math
import os, sys
import numpy as np
import csv
import pickle
import matplotlib.dates as mdates
from sympy import exp

# Dengue surveillance data API configuration
#infodengue_api = "https://api.mosqlimate.org/api/datastore/infodengue/"
#mosqlimate_key = {"X-UID-Key": "DavideNicola96:772df8a3-414a-438a-93b0-f326fea382e9"}
#disease = "dengue"

# Primary model parameters to be estimated via Bayesian inference
#par = {
#    'k_v': 0.466,   # Probability of infection per bite from infected human
#    'k_h': 0.314,   # Probability of infection per bite from infected vector
#    's_0': 0.128    # Proportion of susceptible population
#}

# Weather-dependent parameters for vector biology modeling
dict_weather_coeffs = {
    'A': 0.15,
    'HA': 33256.,
    'HH': 50543.,
    'TH': 301.67,
    'b0': 5.,
    'Emax': 5., #80.,
    'Emean': 7.,
    'Evar': 2.,
    'bite': 0.00161,
}

# Plot style
PLOT_STYLE_PRESENTATION = {
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "figure.titlesize": 22
}

PLOT_STYLE_PAPER = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 10,
    "figure.titlesize": 14
}

#Color scheme and Color map
COLOR_SCHEME = {
    "Purple": {
        "95%": '#e6c6e6',
        "75%": '#cce5ff',
        "50%": '#d8b5d8',
        "line": '#6a51a3'
    },
    "Orange": {
        "95%": '#fddcb5',
        "75%": '#fdbd7a',
        "50%": '#fca044',
        "line": '#c0392b'
    },
    "Red": {
        "95%": '#ffc4c4',
        "75%": '#ff8f8f',
        "50%": '#ff5c5c',
        "line": '#cc0000'
    }
}

STATE_COLOR_MAP = {
    "Amazonas": "Red",
    "Bahia": "Orange",
    "Distrito Federal": "Orange",
    "Goiás": "Orange",
    "Minas Gerais": "Orange",
    "Paraná": "Purple",
    "Rio de Janeiro": "Orange",
    "Rio Grande do Sul": "Red",
    "Santa Catarina": "Purple",
    "São Paulo": "Purple",
}



# Fixed biological parameters based on literature
egg_lper = 0.01      # Percentage of female mosquitoes laying eggs per day
female_per = 0.5     # Proportion of female mosquitoes
mu_v = 0.02941       # Vector mortality rate (1/days)
psi_v = 0.05         # Vertical transmission rate
mu_e = 0.15          # Egg mortality rate (1/days)
alpha_v = 0.1428     # Vector incubation rate (1/days)
pi_h = 1. / 25500.   # Human birth rate (1/days)
mu_h = 1. / 25500.   # Human mortality rate (1/days)
alpha_h = 0.33       # Human incubation rate (1/days)
beta_h = 0.30        # Human recovery rate (1/days)
sigma_h = 0.0001     # Disease-induced mortality rate (1/days)

# Vectors carrying capacity
cc_v = 3

# Minimum value to prevent numerical issues
MIN_VALUE = 1e-6


def fetch_dengue_data_state_from_csv(state_geocodes, start_date, end_date):
    """
    Fetch dengue case data for a Brazilian state from local CSV files.
    
    This function aggregates dengue cases and population data at the state level
    for model training and validation.
    
    Parameters:
    -----------
    state_geocodes : list
        List of municipality geocodes for the target state
    start_date : str
        Start date for data extraction (YYYY-MM-DD format)
    end_date : str
        End date for data extraction (YYYY-MM-DD format)
    target: str
        target_1, target 2 etc
        
    Returns:
    --------
    weekly_cases : pd.DataFrame
        Aggregated weekly dengue cases with population data
    """
    # Load dengue surveillance data
    cases_data = pd.read_csv(
        './data_imdc_2026/dengue.csv',
        parse_dates=['date'],
        usecols=['date', 'epiweek', 'geocode', 'casos', 'train_1', 'target_1']
    )
    
    # Filter data by target state and date range
    cases_data = cases_data[cases_data['geocode'].isin(state_geocodes)]
    date_mask = (cases_data['date'] >= start_date) & (cases_data['date'] <= end_date)
    cases_data = cases_data[date_mask].reset_index(drop=True)

    # Aggregate cases by week (epidemiological surveillance standard)
    cases_data['date'] = pd.to_datetime(cases_data['date'])
    cases_data.set_index('date', inplace=True)
    weekly_cases = cases_data.groupby('date').agg({
        'epiweek': 'first',
        'casos': 'sum',
    }).reset_index()
    weekly_cases.columns = ['data_iniSE',  'epiweek', 'casos']

    # Load population data for normalization
    pop_data = pd.read_csv(
        './data_imdc_2026/datasus_population_2001_2025.csv'
    )
    pop_data['geocode'] = pop_data['geocode'].astype(int)
    
    # Filter and aggregate population data by year
    pop_data = pop_data[pop_data['geocode'].isin(state_geocodes)]
    pop_static = pop_data[pop_data.year == 2021] #need this to compute major cities below
    
    date_mask_year = (pop_data['year'] >= pd.to_datetime(start_date).year) & (pop_data['year'] <= pd.to_datetime(end_date).year)
    pop_data = pop_data[date_mask_year].reset_index(drop=True)
    pop_data_by_year = pop_data.groupby('year').agg({
        'population': 'sum',
    }).reset_index()

    # Merge cases with population data
    weekly_cases['year'] = pd.to_datetime(weekly_cases['data_iniSE']).dt.year
    weekly_cases = weekly_cases.merge(pop_data_by_year, on='year', how='left')
    weekly_cases = weekly_cases[['data_iniSE',  'epiweek', 'casos', 'population']]
    weekly_cases.columns = ['data_iniSE',  'epiweek', 'casos', 'pop']
    weekly_cases = weekly_cases.sort_values('data_iniSE', ascending=True)
    weekly_cases = weekly_cases.reset_index(drop=True)

    # Define 10 major cities in state
    major_cities = pop_static.sort_values('population')[:10].geocode.to_list()

    return weekly_cases, major_cities


def force_sunday(series):
    series = pd.to_datetime(series)

    offset_days = (6 - series.dt.dayofweek) % 7

    return series + pd.to_timedelta(offset_days, unit="D")

def load_or_fetch_cases(state, state_geocodes, start_date, end_date): #XX check if it's better using train-target
    """
    Load cached cases data for a state if available,
    otherwise fetch and cache it.
    Uses 'time' as index.
    """
    cases_state_dir = f"./data_imdc_2026/cases_state/{state}"
    os.makedirs(cases_state_dir, exist_ok=True)
    state_cache_file = os.path.join(cases_state_dir, "cases.csv")

    major_cities_dir = f"./data_imdc_2026/major_cities/{state}"
    os.makedirs(major_cities_dir, exist_ok=True)
    major_cities_cache_file = os.path.join(major_cities_dir, "major.pickle")
    
    if os.path.exists(state_cache_file) and os.path.exists(major_cities_cache_file):
        print(f"Using cached cases data for {state}")
        df = pd.read_csv(state_cache_file, parse_dates=True)

        with open(major_cities_cache_file, 'rb') as f:
            major_cities = pickle.load(f)

    else:
        start_date_fetch = start_date
        end_date_fetch = end_date

        print(f"Fetching cases data for {state}...")
        df_csv, major_cities = fetch_dengue_data_state_from_csv(state_geocodes, start_date_fetch, end_date_fetch)

        df_csv['data_iniSE'] = force_sunday(df_csv['data_iniSE'])
        
        df = df_csv.copy()

        df.to_csv(state_cache_file)
        print(f"Saved cases data to {state_cache_file}")

        with open(major_cities_cache_file, 'wb') as f:
            pickle.dump(major_cities, f)
        
        print(f"Saved major cities data to {major_cities_cache_file}")

    mask = (df['data_iniSE'] >= str(start_date)) & (df['data_iniSE'] <= str(end_date))
    df = df.loc[mask].reset_index(drop=True)
    return df, major_cities


def comp_dew_point(temp, rel_hum):
    """
    Compute dew point in Celsius
    
    :param temp: Air temperature in Celsius (°C)
    :param rel_hum: Relative humidity (es. 60%)
    :return: Temperature of dew point in Celsius (°C)
    """
    # Constants from Magnus-Tetens
    a = 17.625
    b = 243.04
    
    # Gamma (alfa)
    gamma = ((a * temp) / (b + temp)) + math.log(rel_hum / 100.0)
    
    dew_point = (b * gamma) / (a - gamma)
    
    return dew_point


def comp_evaporation(temp, rel_hum, pressure, precipitation):
    """
    Compute evaporation constrained by precipitation and atmospheric conditions.
    """
    # Calculate potential evaporation (your original physical approach)
    es = 6.1078 * math.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (rel_hum / 100.0)
    vpd = es - ea
    gamma_pressure = 0.000662 * pressure
    
    wind_speed = 2.0
    transport_coefficient = 0.26 * (1 + 0.54 * wind_speed)
    pet = (transport_coefficient * vpd) / gamma_pressure
    
    # Adjust potential evaporation based on temperature (energy limitation)
    if temp < 0:
        pet = pet * 0.1  # Frozen conditions limit evaporation
    elif temp < 10:
        pet = pet * (0.3 + 0.07 * temp)  # Low temperatures limit energy
    else:
        # At higher temperatures, energy is usually sufficient
        pet = min(pet, 3.0 + 0.2 * temp)  # Cap at reasonable values
    
    # Constrain by precipitation (water availability)
    if precipitation > 0:
        # In wet conditions, evaporation approaches PET
        # In dry conditions, it's limited by water availability
        if pet > 0:
            ratio = precipitation / pet
            # Logistic function to transition between water-limited and energy-limited
            evap_ratio = 1 - math.exp(-ratio * 1.5)
            evaporation = pet * evap_ratio
        else:
            evaporation = 0
    else:
        # No rain: only minimal evaporation from soil moisture
        evaporation = min(0., pet * 0.1)
    
    # Additional constraint: evaporation cannot exceed precipitation 
    max_evap = precipitation 
    evaporation = min(evaporation, max_evap)
    
    return max(0.0, evaporation)


def Briere(T, a, T0, Tmax):
    """
    Brière temperature response function for arthropod biology.
    
    This nonlinear function captures the temperature dependence of biological
    rates in ectothermic organisms like Aedes aegypti mosquitoes.
    
    Parameters:
    -----------
    T : float/array
        Temperature (°C)
    a : float
        Rate coefficient
    T0 : float
        Lower temperature threshold
    Tmax : float
        Upper temperature threshold
        
    Returns:
    --------
    rate : float/array
        Temperature-dependent biological rate
    """
    T = np.clip(T, T0, Tmax)
    return a * T * (T - T0) * np.sqrt(Tmax - T)

def weather_functions_Aedes(geocodes, start_date, end_date, coeffs):
    """
    Calculate weather-dependent Aedes aegypti biological parameters.
    
    This function transforms meteorological data into biologically meaningful
    parameters for the dengue transmission model:
    - theta: temperature-dependent development rate
    - bite: temperature-dependent biting rate  
    - egg: moisture-dependent egg laying rate
    
    Parameters:
    -----------
    geocodes : str
        Target geocodes
    start_date : datetime
        Start date for weather data
    end_date : datetime
        End date for weather data
    coeffs : dict
        Weather response coefficients
        
    Returns:
    --------
    weather_data : pd.DataFrame
        Daily weather data with calculated biological parameters
    """
    K = 273.15  # Kelvin conversion constant

    A, HA, HH, TH, b0, Emax, Emean, Evar, bite = coeffs.values()

    weather_data = fetch_weather_data(geocodes, start_date, end_date)
    # Smooth temperature with 7-day rolling mean to reduce noise
    weather_data['temp_r'] = weather_data['temp'].rolling(7, min_periods=1).mean()

    # Initialize biological parameter arrays
    Theta = np.zeros(shape = weather_data.shape[0], dtype = np.float64)
    Egg = np.zeros(shape = weather_data.shape[0], dtype = np.float64)
    Bite = np.zeros(shape = weather_data.shape[0], dtype = np.float64)
    Incubation = np.zeros(shape = weather_data.shape[0], dtype = np.float64)
    
    for t in range(weather_data.shape[0]):
        Temp = weather_data['temp_r'].iloc[t]

        # Calculation of development rate
        Theta[t] = A * ((Temp + K) / 298.15) * exp((HA / (1.987)) * (1 / 298.15 - 1 / (Temp + K))) * \
            (1 + exp((HH / (1.987)) * (1 / TH - 1 / (Temp + K))))
            
        # Brière function for biting rate
        Bite[t] = Briere(Temp, 1, 15, 40.08)
        Incubation[t] = Briere(Temp, 1, 10.68, 45.90)

        # Moisture-dependent egg laying (requires 3-day accumulation)
        """if t >= 3:
            Precipitation = 0
            Evaporation = 0
            # Calculate 3-day moisture balance
            for d in range(t - 2, t + 1):
                temp = weather_data['temp'].iloc[d]
                Td = weather_data['dwpt'].iloc[d]

                Precipitation += weather_data['prcp'].iloc[d]
                Evaporation += weather_data['evap'].iloc[d]

            Moisture = Precipitation - Evaporation
            weather_data.loc[weather_data.index[t], 'moisture'] = Moisture"""
            # Sigmoid response to moisture availability
        Egg[t] = b0 + Emax / (1 + math.exp(-(weather_data['rhum'].iloc[t] - Emean) / Evar))

    weather_data['theta'] = Theta
    weather_data['egg'] = Egg
    weather_data['bite'] = Bite
    weather_data['incubation'] = Incubation
    
    # Remove first 2 days (incomplete moisture calculations)
    weather_data = weather_data.iloc[2:,:]
    
    return weather_data
    

def fetch_weather_data(geocodes, start_date, end_date):
    """
    Download meteorological data from nearest weather station.
    
    Retrieves daily temperature, humidity, precipitation, and wind data
    needed for vector biology modeling.
    """
    # Load climate data
    #weather_data = pd.read_csv(
     #   './data_imdc_2026/climate.csv', 
      #  parse_dates=['date'],
       # usecols=['geocode','date','epiweek','temp_med','rel_humid_med','precip_med','pressure_med','dwpt','evap'])

    weather_data = pd.read_parquet(
        './data_imdc_2026/weather_data_daily.parquet', engine='fastparquet',
        columns=['geocode','date','temp_med','rel_humid_med','precip_med','pressure_med'])

    weather_data['geocode'] = weather_data['geocode'].astype(int)
    data = weather_data[weather_data.geocode.isin(geocodes)]   
    data = data[(data.date>start_date) & (data.date<end_date)]     
    data = data[['geocode','date','temp_med', 'rel_humid_med', 'precip_med']]
    data = data.rename(columns={'temp_med':'temp','rel_humid_med':'rhum','precip_med':'prcp'})

    return data


def get_state_weather_data(start_date, end_date, weather_coeffs, state_geocodes):
    """
    Collect and average weather data from the 10 most populated cities in a state.
    
    Weather parameters are crucial for dengue transmission modeling as they
    affect vector biology (development rates, survival, biting behavior).
    
    Parameters:
    -----------
    start_date : str
        Start date for weather data
    end_date : str
        End date for weather data
    weather_coeffs : dict
        Coefficients for weather-dependent biological processes
    state_geocodes : pd.DataFrame
        geocodes to sample weather data from
        
    Returns:
    --------
    avg_weather_data : pd.DataFrame
        State-averaged weather-dependent biological parameters
    """
    
    weather_data_list = []
    successful_cities = []

    # Calculate expected length (number of days in your date range)
    expected_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days - 3
    
    # Collect weather data from multiple cities to reduce spatial bias
    # XX need to take the 10 most populated cities here
    
    for geocode in state_geocodes:
        print(geocode)
        weather_data = weather_functions_Aedes(
            [geocode],
            pd.to_datetime(start_date), 
            pd.to_datetime(end_date), 
            weather_coeffs
        )
    
        # Check if we got data and it has the expected length
        if weather_data is not None and len(weather_data) == expected_length:
            weather_data_list.append(weather_data)
            successful_cities.append(geocode)
            print(f"  ✓ Added {geocode} with {len(weather_data)}/{expected_length} days")
        else:
            if weather_data is None:
                print(f"  ✗ SKIPPED {geocode}: No data returned")
            else:
                print(f"  ✗ SKIPPED {geocode}: Has {len(weather_data)} days, expected {expected_length} days")
    
    # Average weather-dependent parameters across cities
    avg_weather_data = weather_data_list[0].copy()
    for col in ['bite', 'egg', 'theta']:
        if col in avg_weather_data.columns:
            col_data = np.array([df[col].values for df in weather_data_list])
            avg_weather_data[col] = np.mean(col_data, axis=0)
    
    return avg_weather_data


def load_or_fetch_weather(state, start_date, end_date, dict_weather_coeffs, state_geocodes, mode):
    """
    Load cached weather data for a state if available,
    otherwise fetch and cache it.
    Uses 'time' as index.
    """
    
    state_dir = f"./data_imdc_2026/weather_state/{state}"
    os.makedirs(state_dir, exist_ok=True)  # make sure directory exists
    if mode == 'train':
        cache_file = os.path.join(state_dir, "weather.csv")
    elif mode == 'forecast':
        cache_file = os.path.join(state_dir, "weather_forecast.csv")

    if os.path.exists(cache_file):
        print(f"Using cached weather data for {state}")
        df = pd.read_csv(cache_file)
    else:
        print(f"Fetching weather data for {state}...")
        df = get_state_weather_data(
        start_date,
        end_date,
        dict_weather_coeffs,
        state_geocodes
        )
        
        #df = df.reset_index()

        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df = df.loc[mask].reset_index(drop=True)
        
        df.to_csv(cache_file)
        print(f"Saved weather data to {cache_file}")

    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df.loc[mask].reset_index(drop=True)
    return df
