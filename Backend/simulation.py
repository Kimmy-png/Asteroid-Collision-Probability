import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple

COLLISION_THRESHOLD_AU = 0.05 

def run_monte_carlo(
    asteroid_details: pd.DataFrame, 
    num_simulations: int = 10000
) -> Tuple[np.ndarray, float]:
    """
    Menjalankan simulasi Monte Carlo pada MOID asteroid berdasarkan
    skor ketidakpastian (uncertainty) orbitnya.

    Ini adalah pendekatan untuk memperkirakan probabilitas tabrakan
    tanpa memerlukan superkomputer N-body.

    Args:
        asteroid_details: DataFrame satu baris berisi info asteroid yang dipilih.
        num_simulations: Jumlah "klon" virtual yang akan dibuat.

    Returns:
        Tuple: 
        (np.ndarray): Array berisi hasil MOID dari semua klon (untuk plotting).
        (float): Probabilitas tabrakan yang dihitung (0.0 hingga 1.0).
    """
    
    try:
        moid_mean = asteroid_details['min_orbit_intersection'].values[0]
        uncertainty_score = asteroid_details['orbit_uncertainty'].values[0] # Skor 0-9
    except IndexError:
        print("Error: DataFrame asteroid_details kosong.")
        return np.array([]), 0.0
    except KeyError as e:
        print(f"Error: Kolom {e} tidak ditemukan di asteroid_details.")
        return np.array([]), 0.0

    # - Skor 0 -> (0/9)^2 + 0.01 = 0.01 (1% std dev, sangat pasti)
    # - Skor 5 -> (5/9)^2 + 0.01 = 0.31 (31% std dev, sedang)
    # - Skor 9 -> (9/9)^2 + 0.01 = 1.01 (101% std dev, sangat tidak pasti)

    std_dev_multiplier = ((uncertainty_score / 9.0) ** 2) + 0.01
    
    std_dev = moid_mean * std_dev_multiplier

    print(f"Menjalankan {num_simulations} simulasi untuk '{asteroid_details['name'].values[0]}'")
    print(f"MOID Terukur: {moid_mean:.4f} AU")
    print(f"Skor Ketidakpastian: {uncertainty_score}/9 -> Deviasi Standar (σ): {std_dev:.4f} AU")

    simulations = np.random.normal(moid_mean, std_dev, num_simulations)
    
    simulations = np.maximum(0, simulations)
    
    collisions = np.sum(simulations < COLLISION_THRESHOLD_AU)
    
    probability = collisions / num_simulations
    
    print(f"Hasil: {collisions} dari {num_simulations} klon menabrak (Probabilitas: {probability * 100:.6f}%)")
    
    return simulations, probability

def plot_simulation_viz(
    simulation_results: np.ndarray, 
    moid_mean: float,
    asteroid_name: str
) -> go.Figure:
    """
    Membuat histogram Plotly interaktif dari hasil simulasi Monte Carlo.

    Args:
        simulation_results: Array NumPy berisi MOID dari semua klon.
        moid_mean: MOID asli yang terukur (untuk ditandai di grafik).
        asteroid_name: Nama asteroid untuk judul grafik.

    Returns:
        Plotly Figure object.
    """
    
    fig = go.Figure()

    
    fig.add_trace(go.Histogram(
        x=simulation_results, 
        name='Distribusi MOID Klon',
        marker_color='#3b82f6', # Warna biru
        opacity=0.75
    ))

   
    fig.add_vline(
        x=COLLISION_THRESHOLD_AU, 
        line_dash="dash", 
        line_color="#ef4444", 
        line_width=2,
        annotation_text="Batas Tabrakan (0.05 AU)",
        annotation_position="top left"
    )
    
    fig.add_vline(
        x=moid_mean,
        line_dash="solid",
        line_color="#facc15", 
        line_width=3,
        annotation_text="MOID Terukur (Rata-rata)",
        annotation_position="top right"
    )

  
    fig.update_layout(
        title_text=f'Distribusi Probabilitas MOID untuk {asteroid_name} ({len(simulation_results)} klon)',
        xaxis_title="Jarak Minimum dengan Orbit Bumi (MOID dalam AU)",
        yaxis_title="Jumlah Klon Virtual",
        bargap=0.05, 
        legend_title="Legenda",
        plot_bgcolor='rgba(0,0,0,0)' 
    )
    
    return fig