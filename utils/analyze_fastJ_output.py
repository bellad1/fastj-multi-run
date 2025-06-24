#!/usr/bin/env python3
"""
Fast-J Results Plotting and Analysis Module

This module provides functions to load, analyze, and visualize Fast-J results
from saved dask arrays and metadata files.

Usage:
    from analyze_fastJ_output import *

    # Load results
    results, metadata = load_fastj_results('urban_pollution_study')

    # Plot single species profile
    plot_species_profile(results, metadata, 'NO2', run_index=0)

    # Plot all runs for a species
    plot_all_runs_species(results, metadata, 'O3')
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle
from pathlib import Path

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================
def load_fastj_results(run_name, output_dir='fastj_results'):
    """
    Load Fast-J results and metadata.

    Parameters:
    -----------
    run_name : str
        Name of the run to load
    output_dir : str
        Directory containing saved results

    Returns:
    --------
    results : numpy.ndarray or dask.array
        Loaded results array
    metadata : dict
        Loaded metadata
    """
    # Auto-detect file format
    results_file = None
    for ext in ['.npz', '.zarr', '.h5']:
        candidate = os.path.join(output_dir, f"{run_name}_results{ext}")
        if os.path.exists(candidate):
            results_file = candidate
            break

    if results_file is None:
        raise FileNotFoundError(f"No results file found for run '{run_name}' in {output_dir}")

    metadata_file = os.path.join(output_dir, f"{run_name}_metadata.pkl")

    # Load results
    if results_file.endswith('.npz'):
        data = np.load(results_file)
        results = data['results']
    elif results_file.endswith('.zarr'):
        import dask.array as da
        results = da.from_zarr(results_file)
        results = results.compute()  # Convert to numpy for plotting
    elif results_file.endswith('.h5'):
        import h5py
        with h5py.File(results_file, 'r') as f:
            results = f['results'][:]

    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Loaded results: {results.shape}")
    print(f"Output method: {metadata.get('output_flatten_method', 'unknown')}")

    return results, metadata


def get_species_info(metadata):
    """
    Get information about chemical species and data structure.

    Parameters:
    -----------
    metadata : dict
        Results metadata

    Returns:
    --------
    info : dict
        Dictionary with species names, altitudes, and data structure info
    """
    output_meta = metadata.get('output_metadata', {})

    info = {
        'species': output_meta.get('species', []),
        'altitudes': output_meta.get('altitudes', []),
        'n_species': output_meta.get('n_species', 0),
        'n_altitudes': output_meta.get('n_altitudes', 0),
        'output_method': metadata.get('output_flatten_method', 'unknown'),
        'total_runs': metadata.get('total_runs', 0)
    }

    return info


# =============================================================================
# SINGLE SPECIES PLOTTING FUNCTIONS
# =============================================================================

def plot_species_profile(results, metadata, species_name, run_index=0,
                         ax=None, **kwargs):
    """
    Plot vertical profile of j-values for a specific species and run.

    Parameters:
    -----------
    results : numpy.ndarray
        Results array
    metadata : dict
        Results metadata
    species_name : str
        Name of chemical species (e.g., 'NO2', 'O3')
    run_index : int
        Index of run to plot (0 = first run)
    ax : matplotlib.axes, optional
        Axes to plot on
    **kwargs : dict
        Additional arguments passed to plot()

    Returns:
    --------
    fig, ax : matplotlib objects
        Figure and axes objects
    """
    info = get_species_info(metadata)

    # Check if species exists
    if species_name not in info['species']:
        available = ', '.join(info['species'][:10])
        raise ValueError(f"Species '{species_name}' not found. Available: {available}...")

    # Get species index
    species_idx = info['species'].index(species_name)

    # Extract data based on output method
    if info['output_method'] == 'full_2d':
        # 3D array: (n_runs, n_altitudes, n_species)
        if results.ndim != 3:
            raise ValueError(f"Expected 3D array for full_2d method, got {results.ndim}D")

        jvals = results[run_index, :, species_idx]
        altitudes = info['altitudes']
        print(f"jvals = {jvals}")
        print(f"altitudes = {altitudes}")

    elif info['output_method'] in ['surface_only', 'altitude_integrated', 'flatten']:
        # 2D array: (n_runs, n_features)
        if results.ndim != 2:
            raise ValueError(f"Expected 2D array for {info['output_method']} method, got {results.ndim}D")

        if info['output_method'] == 'surface_only':
            # Single value per species
            jval_surface = results[run_index, species_idx]
            print(f"Surface j-value for {species_name}: {jval_surface:.2e} s⁻¹")
            return None, None  # Can't plot profile for surface-only data
        else:
            raise NotImplementedError(f"Plotting not yet implemented for {info['output_method']} method")

    else:
        raise ValueError(f"Unknown output method: {info['output_method']}")

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    # Default plot styling
    plot_kwargs = {
        'linewidth': 2,
        'marker': 'o',
        'markersize': 4,
        'label': f'Run {run_index + 1}'
    }
    plot_kwargs.update(kwargs)

    # Plot profile
    ax.semilogx(jvals, altitudes, **plot_kwargs)

    # Formatting
    ax.set_xlabel(f'J({species_name}) [s⁻¹]', fontsize=12)
    ax.set_ylabel('Altitude [km]', fontsize=12)
    ax.set_title(f'{species_name} Photolysis Rate Profile\n(Run {run_index + 1})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add value range to title
    jval_range = f'{jvals.min():.1e} to {jvals.max():.1e} s⁻¹'
    ax.text(0.02, 0.98, f'Range: {jval_range}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig, ax


def plot_all_runs_species(results, metadata, species_name, max_runs=None,
                          altitude_range=None, colormap='viridis', alpha=0.7):
    """
    Plot vertical profiles for all runs of a specific species.

    Parameters:
    -----------
    results : numpy.ndarray
        Results array
    metadata : dict
        Results metadata
    species_name : str
        Name of chemical species
    max_runs : int, optional
        Maximum number of runs to plot (for readability)
    altitude_range : tuple, optional
        (min_alt, max_alt) to limit altitude range
    colormap : str
        Colormap for different runs
    alpha : float
        Transparency of lines

    Returns:
    --------
    fig, ax : matplotlib objects
        Figure and axes objects
    """
    info = get_species_info(metadata)

    # Check if species exists
    if species_name not in info['species']:
        available = ', '.join(info['species'][:10])
        raise ValueError(f"Species '{species_name}' not found. Available: {available}...")

    # Check if we can plot profiles
    if info['output_method'] != 'full_2d':
        raise ValueError(f"Cannot plot profiles for output method '{info['output_method']}'. Use 'full_2d' method.")

    # Get species index
    species_idx = info['species'].index(species_name)

    # Determine number of runs to plot
    n_runs = results.shape[0]
    if max_runs is not None:
        n_runs = min(n_runs, max_runs)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get colormap
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_runs))

    # Plot each run
    altitudes = info['altitudes']

    for i in range(n_runs):
        jvals = results[i, :, species_idx]

        # Apply altitude range filter if specified
        if altitude_range is not None:
            alt_mask = (altitudes >= altitude_range[0]) & (altitudes <= altitude_range[1])
            jvals_plot = jvals[alt_mask]
            altitudes_plot = altitudes[alt_mask]
        else:
            jvals_plot = jvals
            altitudes_plot = altitudes

        # Plot with different colors
        ax.semilogx(jvals_plot, altitudes_plot,
                    color=colors[i], alpha=alpha, linewidth=1.5,
                    label=f'Run {i+1}' if n_runs <= 10 else None)

    # Formatting
    ax.set_xlabel(f'J({species_name}) [s⁻¹]', fontsize=12)
    ax.set_ylabel('Altitude [km]', fontsize=12)
    ax.set_title(f'{species_name} Photolysis Rate Profiles\n({n_runs} runs)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if altitude_range:
        ax.set_ylim(altitude_range)

    # Add legend only if not too many runs
    if n_runs <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add statistics
    all_jvals = results[:n_runs, :, species_idx]
    jval_mean = np.mean(all_jvals, axis=0)
    jval_std = np.std(all_jvals, axis=0)

    # Plot mean and std envelope
    ax.semilogx(jval_mean, altitudes, 'k-', linewidth=3, label='Mean', alpha=0.8)
    ax.fill_betweenx(altitudes, jval_mean - jval_std, jval_mean + jval_std,
                     color='gray', alpha=0.3, label='±1σ')

    plt.tight_layout()
    return fig, ax


# =============================================================================
# MULTI-SPECIES COMPARISON FUNCTIONS
# =============================================================================

def plot_species_comparison(results, metadata, species_list, run_index=0,
                            altitude_range=None):
    """
    Compare profiles of multiple species for a single run.

    Parameters:
    -----------
    results : numpy.ndarray
        Results array
    metadata : dict
        Results metadata
    species_list : list
        List of species names to compare
    run_index : int
        Index of run to plot
    altitude_range : tuple, optional
        (min_alt, max_alt) to limit altitude range

    Returns:
    --------
    fig, ax : matplotlib objects
        Figure and axes objects
    """
    info = get_species_info(metadata)

    if info['output_method'] != 'full_2d':
        raise ValueError(f"Cannot plot profiles for output method '{info['output_method']}'. Use 'full_2d' method.")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each species
    colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))

    for i, species_name in enumerate(species_list):
        if species_name not in info['species']:
            print(f"Warning: Species '{species_name}' not found, skipping")
            continue

        species_idx = info['species'].index(species_name)
        jvals = results[run_index, :, species_idx]
        altitudes = info['altitudes']

        # Apply altitude range filter if specified
        if altitude_range is not None:
            alt_mask = (altitudes >= altitude_range[0]) & (altitudes <= altitude_range[1])
            jvals_plot = jvals[alt_mask]
            altitudes_plot = altitudes[alt_mask]
        else:
            jvals_plot = jvals
            altitudes_plot = altitudes

        ax.semilogx(jvals_plot, altitudes_plot,
                    color=colors[i], linewidth=2, marker='o', markersize=3,
                    label=f'J({species_name})')

    # Formatting
    ax.set_xlabel('J-values [s⁻¹]', fontsize=12)
    ax.set_ylabel('Altitude [km]', fontsize=12)
    ax.set_title(f'Species Comparison (Run {run_index + 1})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)

    if altitude_range:
        ax.set_ylim(altitude_range)

    plt.tight_layout()
    return fig, ax


def plot_surface_values_all_species(results, metadata, run_index=0, top_n=20):
    """
    Plot surface j-values for all species (bar chart).

    Parameters:
    -----------
    results : numpy.ndarray
        Results array
    metadata : dict
        Results metadata
    run_index : int
        Index of run to plot
    top_n : int
        Number of highest species to show

    Returns:
    --------
    fig, ax : matplotlib objects
        Figure and axes objects
    """
    info = get_species_info(metadata)

    if info['output_method'] == 'full_2d':
        # Get surface values (last altitude level)
        surface_jvals = results[run_index, -1, :]
    elif info['output_method'] == 'surface_only':
        # Already surface values
        surface_jvals = results[run_index, :]
    else:
        raise ValueError(f"Cannot extract surface values for method '{info['output_method']}'")

    # Create DataFrame for easy sorting
    df = pd.DataFrame({
        'species': info['species'][:len(surface_jvals)],
        'j_value': surface_jvals
    })

    # Sort by j-value and take top N
    df_sorted = df.sort_values('j_value', ascending=False).head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.bar(range(len(df_sorted)), df_sorted['j_value'],
                  color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))

    # Formatting
    ax.set_yscale('log')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['species'], rotation=45, ha='right')
    ax.set_ylabel('J-values [s⁻¹]', fontsize=12)
    ax.set_title(f'Surface J-values by Species (Run {run_index + 1})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted['j_value'])):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width()/2, value, f'{value:.1e}',
                    ha='center', va='bottom', rotation=90, fontsize=8)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def quick_species_plot(run_name, species_name, run_index=0, output_dir='fastj_results'):
    """
    Quick function to load data and plot a species profile.

    Parameters:
    -----------
    run_name : str
        Name of saved run
    species_name : str
        Chemical species name
    run_index : int
        Run index to plot
    output_dir : str
        Directory containing results

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    results, metadata = load_fastj_results(run_name, output_dir)
    return plot_species_profile(results, metadata, species_name, run_index)


def quick_all_runs_plot(run_name, species_name, max_runs=50, output_dir='fastj_results'):
    """
    Quick function to load data and plot all runs for a species.

    Parameters:
    -----------
    run_name : str
        Name of saved run
    species_name : str
        Chemical species name
    max_runs : int
        Maximum runs to plot
    output_dir : str
        Directory containing results

    Returns:
    --------
    fig, ax : matplotlib objects
    """
    results, metadata = load_fastj_results(run_name, output_dir)
    return plot_all_runs_species(results, metadata, species_name, max_runs)


def list_available_species(run_name, output_dir='fastj_results'):
    """
    List all available chemical species in a results file.

    Parameters:
    -----------
    run_name : str
        Name of saved run
    output_dir : str
        Directory containing results

    Returns:
    --------
    species : list
        List of available species names
    """
    _, metadata = load_fastj_results(run_name, output_dir)
    info = get_species_info(metadata)

    print(f"Available species ({len(info['species'])}):")
    for i, species in enumerate(info['species']):
        print(f"  {i:2d}: {species}")

    return info['species']


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Example usage
    print("Fast-J Plotting Module")
    print("======================")
    print()
    print("Example usage:")
    print("  from analyze_fastJ_output import *")
    print()
    print("  # Load results")
    print("  results, metadata = load_fastj_results('urban_pollution_study')")
    print()
    print("  # Plot single species profile")
    print("  fig, ax = plot_species_profile(results, metadata, 'NO2', run_index=0)")
    print("  plt.show()")
    print()
    print("  # Plot all runs for a species")
    print("  fig, ax = plot_all_runs_species(results, metadata, 'O3', max_runs=20)")
    print("  plt.show()")
    print()
    print("  # Quick plots")
    print("  quick_species_plot('urban_pollution_study', 'NO2')")
    print("  quick_all_runs_plot('urban_pollution_study', 'O3')")
    print()
    print("  # List available species")
    print("  species = list_available_species('urban_pollution_study')")
