# 1. Team and Contributors  
### Team Name: ISI Dengue

### Members  
* **Davide Nicola** – Junior Researcher, ISI Foundation
* **Daniela Paolotti** – Senior Researcher, ISI Foundation 
* **Mattia Mazzoli** – Senior Researcher, ISI Foundation

---

# 2. Repository Structure  
<pre>
3rd_imdc_isi_isi-dengue/
│
├── Demo Notebooks/             # Demo notebooks inherited from the template repo
│   
├── img/                        # Demo images for the challenge template
│
└── model/                                                  # ISI-Dengue model for the 3rd IMDC
    ├── utils.py                                            # basic functions for extraction of cases and weather variables
    ├── model_utils.py                                      # basic functions for computation of weather-based vectors behavior and demography
    ├── forecast_plot.py                                    # optional function for plotting forecasted epidemic curves
    ├── Sprint_2026_Forecast.py                             # main code, ODE solver, calibration, forecast and upload of results
    ├── Sprint_2026_Prepare_daily_weather.ipynb             # weather-data processing, from weekly to daily reconstructed trends by geocode
    └── Sprint_2026_Prepare_daily_weather-forecast.ipynb    # merge daily weather-data from forecasted weather-data by geocode 
  
</pre>

---

# 3. Libraries and Dependencies  
TODO: All Python dependencies are listed in `requirements.txt`. Key packages include:

- `numpy`, `pandas`, `scipy` (for basic data handling)
- `matplotlib`, `arviz`  (for visualization)
- `pymc` (for Bayesian modeling)  
- `numba` (for accelerating ODE computations)  
- `geopandas` (for spatial operations)  
- `corner`, `pickle`, `json` (supporting utilities)  

---

# 4. Data and Variables  

### Datasets Used  
We used these datasets provided by the Infodengue-Mosqlimate sprint organisers:
- **dengue.csv.gz** – Weekly dengue cases by municipality  
- **datasus_population_2001_2025.gz** – Population data for normalization  
- **shape_muni.gpkg** – Municipality shapefile with names and geo‑codes for data mapping
- **climate.csv.gz** – Temperature and precipitation variables from 1999

### Preprocessing Steps  
- Aggregated weekly dengue case counts from municipality and matched with yearly population by state  
- Selected top 10 most populous cities per state for State weather sampling  
- Computed rolling mean temperature, moisture balance, and derived biological parameters:
  - `theta`: Eggs development rate
  - `egg`: Eggs laying rate,
  - `bite`: Daily bite rate
- Time alignment of epidemiological and weather series ensured consistent daily-weekly mapping  

---

# 5. Model Training  

### Model Architecture  
- A vector–host SEIR ODE system for humans and mosquitoes  
- Uses temperature- and humidity-dependent parameters  
- Implemented with Runge–Kutta (RK4) integration accelerated using Numba  

### Training Procedure  
- Bayesian inference via PyMC with `DEMetropolisZ` sampler  
- Priors: Beta distributions for vector and host infection probabilities  
- Likelihood: Negative Binomial on reported cases  
- Posterior summaries (medians, intervals) are extracted and saved  

---

# 6. References  

- PyMC documentation: https://www.pymc.io/  

---

# 7. Data Usage Restriction  

- **Training** uses only data up to **EW 25** of each year, starting from **EW 41** of the previous year
- **Forecasting** is done from **EW 41** of the current year through **EW 40** of the following year
- This split is enforced in:
  - `fit_function()`
  - `forecast_function()`

---

# VIII. Predictive Uncertainty  

- Posterior sampling via PyMC is used to propagate uncertainty  
- We simulate weekly incidence for each posterior sample  
- Credible intervals (50%, 80%, 90%, 95%) are computed using NumPy percentiles  
- These bounds are returned in forecast CSVs inside the `forecasts/` directory  

---

Original model from the previous IMDC: [https://github.com/DavideNicola/ISI_Dengue_Model?tab=readme-ov-file](https://github.com/DavideNicola/ISI_Dengue_Model)

