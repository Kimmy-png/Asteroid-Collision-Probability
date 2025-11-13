import requests
import pandas as pd
import joblib
import json
from typing import List, Dict, Tuple, Any

# --- FUNGSI 1: Memuat Artefak Model ---
def load_model_artifacts(model_path: str, scaler_path: str, metadata_path: str) -> Tuple[Any, Any, Dict]:
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except Exception as e:
        print(f"❌ Error saat memuat model: {e}")
        return None, None, None

# --- FUNGSI 2: Mengambil Data dari NASA ---
def fetch_nasa_data(api_key: str, pages: int = 3) -> pd.DataFrame:
    asteroids_list = []
    url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={api_key}"

    for page in range(pages):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                break
            
            data = response.json()
            neos = data['near_earth_objects']

            for neo in neos:
                orb = neo['orbital_data']
                features = {
                    'id': neo['id'],
                    'name': neo['name'],
                    'absolute_magnitude': neo.get('absolute_magnitude_h', 0.0),
                    'eccentricity': float(orb.get('eccentricity', 0)),
                    'semi_major_axis': float(orb.get('semi_major_axis', 0)),
                    'inclination': float(orb.get('inclination', 0)),
                    'min_orbit_intersection': float(orb.get('minimum_orbit_intersection', 0)),
                    'jupiter_tisserand_invariant': float(orb.get('jupiter_tisserand_invariant', 0)),
                    'mean_anomaly': float(orb.get('mean_anomaly', 0)),
                    'arg_of_perihelion': float(orb.get('perihelion_argument', 0)), 
                    'orbit_uncertainty': int(orb.get('orbit_uncertainty', 5)),
                }
                
                if neo.get('close_approach_data') and len(neo['close_approach_data']) > 0:
                    vel_kms = neo['close_approach_data'][0]['relative_velocity']['kilometers_per_second']
                    features['relative_velocity_kms'] = float(vel_kms)
                else:
                    features['relative_velocity_kms'] = 0.0

                asteroids_list.append(features)
            
            url = data['links'].get('next')
            if not url:
                break
        except Exception as e:
            print(f"❌ Error koneksi NASA: {e}")
            break

    return pd.DataFrame(asteroids_list)

# --- FUNGSI 3: Filter Data dengan XGBoost ---
def apply_xgboost_filter(df: pd.DataFrame, model: Any, scaler: Any, metadata: Dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    try:
        feature_columns = metadata['feature_columns']
        threshold = metadata['optimal_threshold']
    except KeyError:
        return pd.DataFrame()
    
    df_features = df.copy()
    for col in feature_columns:
        if col not in df_features.columns:
            df_features[col] = 0.0
            
    df_features = df_features[feature_columns].fillna(df_features[feature_columns].median())
    df_scaled = scaler.transform(df_features)
    probabilities = model.predict_proba(df_scaled)[:, 1]
    
    high_risk_mask = probabilities >= threshold
    
    filtered_df = df.loc[high_risk_mask].copy()
    filtered_df['risk_probability'] = probabilities[high_risk_mask]
    
    return filtered_df.sort_values(by='risk_probability', ascending=False)