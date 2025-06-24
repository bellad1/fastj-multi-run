#!/usr/bin/env python3
"""
Fast-J Run Conditions Tracker

This module adds functionality to store and retrieve the exact aerosol
conditions for each run hash, enabling you to trace back from hash to
input parameters.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime


def get_run_conditions(run_hash, conditions_file='run_conditions.json'):
    """
    Retrieve aerosol conditions for a specific run hash.

    Parameters:
    -----------
    run_hash : str
        Run hash to look up
    conditions_file : str
        JSON file containing conditions

    Returns:
    --------
    conditions : dict or None
        Dictionary with run conditions, or None if not found
    """
    if not os.path.exists(conditions_file):
        return None

    with open(conditions_file, 'r') as f:
        conditions_db = json.load(f)

    return conditions_db.get(run_hash)


def list_all_run_conditions(conditions_file='run_conditions.json',
                            output_format='dataframe'):
    """
    List conditions for all runs.

    Parameters:
    -----------
    conditions_file : str
        JSON file containing conditions
    output_format : str
        'dataframe', 'dict', or 'summary'

    Returns:
    --------
    conditions : pandas.DataFrame, dict, or str
        Run conditions in requested format
    """
    if not os.path.exists(conditions_file):
        print(f"No conditions file found: {conditions_file}")
        return None

    with open(conditions_file, 'r') as f:
        conditions_db = json.load(f)

    if output_format == 'dict':
        return conditions_db

    elif output_format == 'dataframe':
        # Convert to DataFrame for easy analysis
        rows = []
        for run_hash, conditions in conditions_db.items():
            row = {
                'hash': run_hash,
                'run_index': conditions.get('run_index', -1),
                'height_m': conditions.get('height_m', 0),
                'aod_550nm': conditions.get('aod_550nm', 0),
                'ssa_550nm': conditions.get('ssa_550nm', 0),
                'g_550nm': conditions.get('g_550nm', 0),
                'aod_mean': conditions.get('aod_mean', 0),
                'ssa_mean': conditions.get('ssa_mean', 0),
                'g_mean': conditions.get('g_mean', 0),
                'timestamp': conditions.get('timestamp', '')
            }
            rows.append(row)

        return pd.DataFrame(rows)

    elif output_format == 'summary':
        summary = "Run Conditions Summary\n"
        summary += "=====================\n"
        summary += f"Total runs: {len(conditions_db)}\n\n"

        for run_hash, conditions in list(conditions_db.items())[:10]:  # Show first 10
            summary += f"Hash: {run_hash}\n"
            summary += f"  Height: {conditions.get('height_m', 0):.1f} m\n"
            summary += f"  AOD(550nm): {conditions.get('aod_550nm', 0):.4f}\n"
            summary += f"  SSA(550nm): {conditions.get('ssa_550nm', 0):.4f}\n"
            summary += f"  G(550nm): {conditions.get('g_550nm', 0):.4f}\n\n"

        if len(conditions_db) > 10:
            summary += f"... and {len(conditions_db) - 10} more runs\n"

        return summary


def find_runs_by_conditions(height_range=None, aod_range=None, ssa_range=None,
                            g_range=None, conditions_file='run_conditions.json'):
    """
    Find runs that match specific aerosol conditions.

    Parameters:
    -----------
    height_range : tuple, optional
        (min_height, max_height) in meters
    aod_range : tuple, optional
        (min_aod, max_aod) at 550nm
    ssa_range : tuple, optional
        (min_ssa, max_ssa) at 550nm
    g_range : tuple, optional
        (min_g, max_g) at 550nm
    conditions_file : str
        JSON file containing conditions

    Returns:
    --------
    matching_hashes : list
        List of run hashes that match criteria
    """
    df = list_all_run_conditions(conditions_file, 'dataframe')

    if df is None or df.empty:
        return []

    # Apply filters
    mask = pd.Series([True] * len(df))

    if height_range:
        mask &= (df['height_m'] >= height_range[0]) & (df['height_m'] <= height_range[1])

    if aod_range:
        mask &= (df['aod_550nm'] >= aod_range[0]) & (df['aod_550nm'] <= aod_range[1])

    if ssa_range:
        mask &= (df['ssa_550nm'] >= ssa_range[0]) & (df['ssa_550nm'] <= ssa_range[1])

    if g_range:
        mask &= (df['g_550nm'] >= g_range[0]) & (df['g_550nm'] <= g_range[1])

    matching_df = df[mask]
    return matching_df['hash'].tolist()


def plot_conditions_space(conditions_file='run_conditions.json'):
    """
    Plot the parameter space covered by all runs.

    Parameters:
    -----------
    conditions_file : str
        JSON file containing conditions

    Returns:
    --------
    fig : matplotlib.figure
        Figure with parameter space plots
    """
    import matplotlib.pyplot as plt

    df = list_all_run_conditions(conditions_file, 'dataframe')

    if df is None or df.empty:
        print("No conditions data available")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # AOD vs Height
    axes[0, 0].scatter(df['aod_550nm'], df['height_m'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('AOD at 550nm')
    axes[0, 0].set_ylabel('Height [m]')
    axes[0, 0].set_title('AOD vs Height')
    axes[0, 0].grid(True, alpha=0.3)

    # SSA vs AOD
    axes[0, 1].scatter(df['aod_550nm'], df['ssa_550nm'], alpha=0.6, s=20, c=df['height_m'], cmap='viridis')
    axes[0, 1].set_xlabel('AOD at 550nm')
    axes[0, 1].set_ylabel('SSA at 550nm')
    axes[0, 1].set_title('SSA vs AOD (colored by height)')
    axes[0, 1].grid(True, alpha=0.3)

    # G vs SSA
    axes[1, 0].scatter(df['ssa_550nm'], df['g_550nm'], alpha=0.6, s=20, c=df['aod_550nm'], cmap='plasma')
    axes[1, 0].set_xlabel('SSA at 550nm')
    axes[1, 0].set_ylabel('G at 550nm')
    axes[1, 0].set_title('G vs SSA (colored by AOD)')
    axes[1, 0].grid(True, alpha=0.3)

    # Height distribution
    axes[1, 1].hist(df['height_m'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Height [m]')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Height Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Example usage functions
def lookup_hash(run_hash):
    """Quick lookup function."""
    conditions = get_run_conditions(run_hash)
    if conditions:
        print(f"Run Hash: {run_hash}")
        print(f"  Height: {conditions['height_m']:.1f} m")
        print(f"  AOD(all): {conditions['aod_values']}")
        print(f"  SSA(all): {conditions['ssa_values']}")
        print(f"  G(all): {conditions['g_values']}")
        print(f"  AOD(550nm): {conditions['aod_550nm']:.4f}")
        print(f"  SSA(550nm): {conditions['ssa_550nm']:.4f}")
        print(f"  G(550nm): {conditions['g_550nm']:.4f}")
        print(f"  Run index: {conditions['run_index']}")
        print(f"  Timestamp: {conditions['timestamp']}")
    else:
        print(f"Hash {run_hash} not found in conditions database")


if __name__ == "__main__":
    print("Fast-J Run Conditions Tracker")
    print("=============================")
    print()
    print("Usage examples:")
    print("  # Look up a specific hash")
    print("  lookup_hash('69ec6eea3727')")
    print()
    print("  # Get all conditions as DataFrame")
    print("  df = list_all_run_conditions(output_format='dataframe')")
    print()
    print("  # Find runs with specific conditions")
    print("  hashes = find_runs_by_conditions(height_range=(1000, 3000), aod_range=(0.1, 0.5))")
    print()
    print("  # Plot parameter space")
    print("  fig = plot_conditions_space()")
