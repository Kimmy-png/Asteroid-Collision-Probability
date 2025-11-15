# Asteroid Collision Probability

A machine learning-powered system that identifies potentially hazardous asteroids and calculates collision probability using AI filtering and Monte Carlo physics simulations.

## Project Overview

This project combines **XGBoost machine learning** with **Monte Carlo physics simulations** to:
1. **Filter asteroids** - Uses a trained XGBoost model to identify potentially hazardous Near-Earth Objects (NEOs) from NASA data
2. **Calculate collision probability** - Runs Monte Carlo simulations based on orbital uncertainty to estimate Earth collision risk
3. **Visualize results** - Interactive dashboard built with Streamlit showing real-time risk assessments

### Key Features
- **Live NASA Data Integration** - Fetches real-time asteroid orbital data via NASA API
- **XGBoost AI Filtering** - Pre-filters candidates based on orbital characteristics
- **Monte Carlo Simulation** - Physics-based probabilistic collision estimation
- **Interactive Visualization** - Streamlit web interface with risk metrics and charts
- **Configurable Parameters** - Adjust simulation count, data pages, and thresholds

---

## Project Structure

```
Asteroid_collision_probability/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── readme.md                       # This file
│
├── Backend/
│   ├── __init__.py
│   ├── data_manager.py            # NASA API integration & XGBoost filtering
│   └── simulation.py              # Monte Carlo collision probability engine
│
├── Dataset/
│   └── asteroid_orbital_data.csv  # Training dataset (hazardous asteroid labels)
│
├── Models/
│   └── model_metadata.json        # Model threshold & feature columns config
│
└── Training/
    └── training.py                # XGBoost model training pipeline
```

---

## Try It Out

Want to see this project in action? Check out the live Streamlit app:

[[Launch the Asteroid Collision Probability]](https://asteroid-collision-probability.streamlit.app/)

---

## How It Works

### 1. **AI-Based Filtering (XGBoost)**

The `data_manager.py` module grabs asteroid data from NASA and runs it through an XGBoost classifier:

**Input Features:**
- `absolute_magnitude` - How bright/big the asteroid is
- `eccentricity` - How oval the orbit is (0 = circular, 1 = parabolic)
- `semi_major_axis` - Average distance from the sun (in AU)
- `inclination` - Angle of the orbit relative to Earth
- `min_orbit_intersection` (MOID) - Closest distance to Earth's orbit
- `jupiter_tisserand_invariant` - Orbital similarities and dynamics
- `mean_anomaly`, `arg_of_perihelion` - Where the asteroid is in its orbit
- `relative_velocity_kms` - How fast it's coming at us

**How it works:**
```python
raw_data = fetch_nasa_data(api_key)  # Get data from NASA
filtered = apply_xgboost_filter(raw_data, model, scaler)  # Filter & classify
```

### 2. **Monte Carlo Physics Simulation**

The `simulation.py` module calculates the actual collision probability:

1. **Model uncertainty** - Takes the uncertainty score (0-9) and turns it into a standard deviation:
   ```
   σ = MOID × [(uncertainty/9)² + 0.01]
   ```

2. **Create virtual clones** - Makes N copies of the asteroid with MOID values pulled from a normal distribution

3. **Check for hits** - Counts how many clones end up with MOID < 0.05 AU (collision zone)

4. **Do the math** - P(collision) = number of hits / total clones

**Example:**
- Say you have an asteroid with MOID=0.1 AU, uncertainty=5/9
- σ = 0.1 × [(5/9)² + 0.01] = 0.0331 AU
- Run 10,000 simulations: 2 clones hit Earth → **0.02% probability**

### 3. **Interactive Visualization**

Plotly creates histograms showing:
- All the simulated MOID values
- Collision threshold line (red) at 0.05 AU
- Measured MOID (yellow line)

---

## Technical Stack

**Machine Learning:**
- **XGBoost** - Gradient boosting for hazard classification
- **Scikit-learn** - Feature scaling and model evaluation
- **Joblib** - Model persistence

**Simulation & Analysis:**
- **NumPy** - Numerical computations for Monte Carlo
- **Pandas** - Data manipulation and analysis

**Frontend & Data:**
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive visualizations
- **NASA NEO API** - Real-time asteroid data

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| Risk Probability (AI Score) | XGBoost confidence (0-1) that asteroid is hazardous |
| Orbit Uncertainty | NASA confidence in orbit calculation (0-9, lower is better) |
| MOID | Minimum Orbit Intersection Distance (AU) - closest approach |
| Collision Threshold | 0.05 AU - if MOID < this, potential collision |
| Relative Velocity | Approach speed in km/s |

---

## Data Sources

- **NASA NEO API**: https://api.nasa.gov/neo/rest/v1/neo/browse - Live near-Earth object orbital data
- **Training Dataset**: Labeled asteroid orbital data with hazard classifications

---

## License

This project is provided as-is for educational and research purposes.
