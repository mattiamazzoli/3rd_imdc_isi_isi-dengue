# 1. Team and Contributors  
### Team Name: ISI Foundation

### Members  
* **Davide Nicola** – Junior Researcher, ISI Foundation
* **Daniela Paolotti** – Senior Researcher, ISI Foundation 
* **Mattia Mazzoli** – Senior Researcher, ISI Foundation 

---

# 2. Repository Structure  
<pre>
TBD
</pre>

---

# 3. Libraries and Dependencies  
TODO: All Python dependencies are listed in `requirements.txt`. Key packages include:

- `numpy`, `pandas`, `scipy`, `matplotlib`, `arviz`  
- `pymc` (for Bayesian modeling)  
- `numba` (for accelerating ODE computations)  
- `meteostat`, `geopy`, `requests` (for weather and geolocation APIs)  
- `corner`, `pickle`, `json`, `odfpy` (supporting utilities)  

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
- Uses temperature-dependent parameters  
- Implemented with Runge–Kutta (RK4) integration accelerated using Numba  

### Training Procedure  
- Bayesian inference via PyMC with `DEMetropolisZ` sampler  
- Priors: Beta distributions for vector and host infection probabilities  
- Likelihood: Negative Binomial on reported cases  
- Posterior summaries (medians, intervals) are extracted and saved  

---

# 6. References  

- PyMC documentation: https://www.pymc.io/  
- Meteostat API docs: https://dev.meteostat.net/  

---

# 7. Data Usage Restriction  

- **Training** uses only data up to **EW 25** of each year, starting from **EW 41** of the previous year
- **Forecasting** is done from **EW 41** of the current year through **EW 40** of the following year
- This split is enforced in:
  - `fit_function()`
  - `forcast_function()`

---

# VIII. Predictive Uncertainty  

- Posterior sampling via PyMC is used to propagate uncertainty  
- We simulate weekly incidence for each posterior sample  
- Credible intervals (50%, 80%, 90%, 95%) are computed using NumPy percentiles  
- These bounds are returned in forecast CSVs inside the `forecasts/` directory  

---

