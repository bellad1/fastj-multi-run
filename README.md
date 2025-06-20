# Fast-J Multi-Run Atmospheric Photochemistry Model

A comprehensive Python-based system for running multiple Fast-J radiative transfer calculations with aerosol property data from ACTIVATE campaign measurements.

## Overview

This repository contains tools for:
- Processing aerosol optical properties from RSP (Research Scanning Polarimeter) and HSRL (High Spectral Resolution Lidar) measurements
- Running Fast-J photolysis rate calculations with varying aerosol conditions
- Managing large-scale computational campaigns with checkpointing
- Analyzing and storing results efficiently using Dask arrays

## Features

- **Multi-run capabilities**: Process thousands of aerosol scenarios automatically
- **Checkpointing**: Resume interrupted calculations
- **Memory efficiency**: Uses Dask for large dataset handling
- **Multiple output formats**: Zarr, HDF5, NumPy support
- **Incremental saving**: Save results periodically during long runs
- **Data validation**: Hash-based run identification and duplicate prevention

## Requirements

### Core Dependencies
```pip install numpy scipy matplotlib pandas dask

### Optional Dependencies (for enhanced functionality)
bashpip install zarr h5py  # For efficient large array storage

### System Requirements
Fast-J executable (compiled from Fortran source)
Python 3.7+
Sufficient disk space for results (varies by campaign size)

## Quick Start
1. Data Preparation
Ensure your CSV files are organized as:
aerosol_fastJcsvFiles/
├── AOD_Urban_Pollution.csv
├── SSA_Urban_Pollution.csv
└── G_Urban_Pollution.csv

aerosol_heights/
└── HSRL_Urban_Pollution_hts.csv
2. Basic Usage
pythonfrom multiRunFastJ import main

### Run with default configuration
results_array, metadata = main()
3. Custom Configuration
Edit the configuration section in multiRunFastJ.py:
python# Number of runs (None = all available)
max_runs = 100

### Output processing
store_mean_intensity = False
output_method = 'surface_only'  # or 'full_2d', 'altitude_integrated'

### File management
append_to_existing = True
save_format = 'zarr'  # or 'numpy', 'hdf5'
File Structure
├── multiRunFastJ.py           # Main multi-run script
├── collocationTest_multiFile.py  # Data collocation and processing
├── AOD_Urban_Pollution.csv    # Aerosol optical depth data
├── HSRL_Urban_Pollution_hts.csv  # Aerosol layer heights
├── jVals.dat                  # Example Fast-J output
├── meanIntensity.dat          # Example mean intensity output
├── README.md                  # This file
└── requirements.txt           # Python dependencies

## Data Input Format
### Aerosol Properties
CSV files with wavelengths in header row and measurements in subsequent rows:
csv187.0,191.0,193.0,196.0,...
0.123,0.145,0.156,0.167,...
0.098,0.112,0.125,0.134,...
### Height Data
Single column CSV with header:
csvHSRL Aerosol Heights for Urban/Pollution
2450.0
1890.0
3120.0
### Output Data
J-values
Photolysis rates for atmospheric species at different altitudes:

Surface-only: J-values at ground level
Full 2D: Complete altitude-species matrix
Altitude-integrated: Column-integrated values

### Storage Formats
Zarr: Recommended for large datasets
NumPy: Simple .npz format
HDF5: Industry standard scientific format

### Configuration Options
#### Processing Methods

surface_only: Extract ground-level photolysis rates
full_2d: Preserve complete altitude×species arrays
altitude_integrated: Integrate over atmospheric column
flatten: Simple 1D flattening

#### Checkpointing
Automatic resume capability using fastj_checkpoint.json:
json{
  "completed_runs": ["hash1", "hash2", ...],
  "saved_runs": ["hash1", "hash2", ...]
}
#### Performance Optimization
Memory Management

Uses Dask for out-of-core computation
Configurable chunking for large arrays
Incremental saving to prevent data loss

Computational Efficiency

Duplicate run detection via hashing
Parallel processing capability (future enhancement)
Optimized I/O for large campaigns

Example Workflows
1. Small Test Run
python# Test with 10 cases
max_runs = 10
output_method = 'surface_only'
save_format = 'numpy'
2. Production Campaign
python# Full dataset processing
max_runs = None  # All available
output_method = 'full_2d'
save_format = 'zarr'
incremental_save = True
save_every = 50
3. Resume Interrupted Run
python# Automatically resumes from checkpoint
append_to_existing = True
fixed_run_name = 'Urban_Pollution'
Results Analysis
Loading Saved Results
pythonfrom multiRunFastJ import load_results

# Load by run name
results, metadata = load_results(run_name='Urban_Pollution_20241201')

# Quick analysis
from multiRunFastJ import analyze_results_quick
analysis = analyze_results_quick(results, metadata)
Available Tools

list_saved_runs(): Browse available datasets
debug_saved_files(): Troubleshoot file issues
analyze_results_quick(): Basic statistical analysis

# Troubleshooting
Common Issues

Missing executable: Ensure fastJX is compiled and executable
Memory errors: Reduce chunk sizes or use incremental saving
File format errors: Check CSV structure and headers
Checkpoint corruption: Delete and restart from clean state

# Debug Functions
python# Check saved files
from multiRunFastJ import debug_saved_files
debug_saved_files('fastj_results')

## List available runs
from multiRunFastJ import list_saved_runs
runs = list_saved_runs()
Contributing

Fork the repository
Create a feature branch (git checkout -b feature/new-analysis)
Commit changes (git commit -am 'Add new analysis method')
Push to branch (git push origin feature/new-analysis)
Create Pull Request

Citation
If you use this code in your research, please cite:
[Your paper citation here]
License
[Specify your license here - MIT, GPL, etc.]
Contact

Author: [Your name]
Email: [Your email]
Institution: [Your institution]

Acknowledgments

ACTIVATE campaign for aerosol measurement data
Fast-J development team
NASA Goddard Space Flight Center


# Fast-J Multi-Run Analysis System

A comprehensive Python framework for running multiple Fast-J photolysis rate calculations with varying aerosol properties, featuring automatic checkpointing, result storage, and analysis tools.

## 📋 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Configuration Required](#user-configuration-required)
- [File Structure](#file-structure)
- [Main Scripts](#main-scripts)
- [Analysis and Plotting](#analysis-and-plotting)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## 🔬 Overview

This code automates the process of running Fast-J photolysis rate calculations across multiple aerosol scenarios. It reads aerosol optical properties (AOD, SSA, asymmetry parameter) and layer heights from CSV files, generates Fast-J input files, executes the model, and stores results efficiently for analysis.

### Key Features
- **Automated Multi-Run Processing**: Handle thousands of aerosol scenarios
- **Hash-Based Checkpointing**: Resume interrupted runs without losing progress
- **Flexible Data Storage**: Save results in numpy, zarr, or HDF5 formats
- **Run Condition Tracking**: Trace back from results to input parameters
- **Comprehensive Plotting Tools**: Visualize j-value profiles and parameter spaces
- **Memory Efficient**: Uses dask arrays for large datasets

## 🖥️ System Requirements

### Software Dependencies
- **Python 3.7+**
- **Fast-J executable** (compiled and working)
- **Required Python packages**:
  ```bash
  pip install numpy pandas dask matplotlib seaborn
  ```
- **Optional packages** (for enhanced performance):
  ```bash
  pip install zarr h5py  # For efficient large array storage
  ```

### Hardware Recommendations
- **RAM**: 8GB+ (for large multi-run studies)
- **Storage**: Sufficient space for results (varies by number of runs)
- **CPU**: Multi-core recommended for faster processing

## 🚀 Installation

1. **Clone or download** this repository to your working directory
2. **Install dependencies**:
   ```bash
   pip install numpy pandas dask matplotlib seaborn zarr h5py
   ```
3. **Ensure Fast-J is compiled** and the executable is available
4. **Prepare your input data** (see [User Configuration Required](#user-configuration-required))

## ⚡ Quick Start

### 1. Basic Single Run
```python
# Set up for testing with 1 case
python multiRunFastJ.py
```

### 2. Production Multi-Run
```python
# Edit configuration in main():
max_runs = 1000              # Number of cases to process
append_to_existing = True    # Combine results in one file
fixed_run_name = 'my_study'  # Consistent filename

# Run the script
python multiRunFastJ.py
```

### 3. Analyze Results
```python
from fastj_plotting import *

# Load and plot results
results, metadata = load_fastj_results('my_study')
quick_species_plot('my_study', 'NO2')
quick_all_runs_plot('my_study', 'O3')
```

## ⚙️ User Configuration Required

### 🔧 **Critical: File Paths and Names**
You **MUST** update these paths in `multiRunFastJ.py`:

```python
# In main() function - UPDATE THESE PATHS:
csvDir_aerosolProps = '/YOUR/PATH/TO/aerosol_fastJcsvFiles'
csvDir_aerosolHeight = '/YOUR/PATH/TO/aerosol_heights'

# UPDATE THESE FILENAMES to match your data:
csvName_AOD = 'AOD_YourAerosolType.csv'
csvName_SSA = 'SSA_YourAerosolType.csv' 
csvName_G = 'G_YourAerosolType.csv'
csvName_height = 'HSRL_YourAerosolType_hts.csv'

# UPDATE EXECUTABLE PATH:
executable_path = './fastJX'  # Path to your Fast-J executable
```

### 📊 **Required Input Data Format**

#### **Aerosol Property Files** (AOD, SSA, G):
```csv
187.0,191.0,193.0,196.0,202.0,208.0,...    # Wavelengths (nm) - header row
0.147,0.146,0.146,0.145,0.144,0.143,...    # Measurement 1
0.107,0.107,0.106,0.106,0.105,0.105,...    # Measurement 2
0.399,0.396,0.395,0.394,0.390,0.387,...    # Measurement 3
...
```

#### **Height File**:
```csv
HSRL Aerosol Heights for YourType          # Header
2450.5                                      # Height 1 (meters)
1876.2                                      # Height 2 (meters)
3021.8                                      # Height 3 (meters)
...
```

### 🎛️ **Configuration Options in main()**

```python
# PROCESSING OPTIONS
max_runs = 1                    # 1=test, 100=chunk, None=all
store_mean_intensity = False    # True=save mean intensity data
output_method = 'surface_only'  # 'surface_only', 'full_2d', etc.

# FILE MANAGEMENT
append_to_existing = True              # Combine results vs separate files
fixed_run_name = 'urban_pollution'     # Consistent filename
checkpoint_file = 'fastj_checkpoint.json'
```

### 🏗️ **Required Fast-J Setup**
1. **Fast-J executable** must be compiled and named `fastJX` (or update path)
2. **CTM_atmo.dat** must exist in working directory
3. **Executable permissions**: `chmod +x fastJX`

## 📁 File Structure

```
your_project/
├── multiRunFastJ.py           # Main script
├── fastj_plotting.py          # Analysis and plotting module
├── run_conditions_tracker.py  # Optional: advanced condition tracking
├── fastJX                     # Fast-J executable
├── CTM_atmo.dat              # Required Fast-J input
├── aerosol_fastJcsvFiles/    # Your aerosol property data
│   ├── AOD_Urban_Pollution.csv
│   ├── SSA_Urban_Pollution.csv
│   └── G_Urban_Pollution.csv
├── aerosol_heights/          # Your height data
│   └── HSRL_Urban_Pollution_hts.csv
├── fastj_results/            # Generated results (auto-created)
│   ├── my_study_results.npz
│   ├── my_study_metadata.pkl
│   └── my_study_summary.txt
├── fastj_checkpoint.json     # Progress tracking (auto-created)
└── run_conditions.json       # Condition database (auto-created)
```

## 🎯 Main Scripts

### `multiRunFastJ.py`
**Primary execution script** that:
- Reads aerosol property CSV files
- Generates Fast-J input files for each combination
- Executes Fast-J
- Stores results with hash-based checkpointing
- Handles interrupted runs gracefully

**Key functions:**
- `main()`: Configure and run everything
- `run_multi_fastj_cases()`: Core multi-run logic
- `save_results()`: Store results with append capability

### `fastj_plotting.py`
**Analysis and visualization module** that:
- Loads saved results automatically
- Creates publication-quality plots
- Handles different output formats (surface-only, full-2D)
- Provides species comparison tools

**Key functions:**
- `load_fastj_results()`: Load saved data
- `plot_species_profile()`: Single species vertical profiles
- `plot_all_runs_species()`: Multi-run variability analysis
- `quick_species_plot()`: One-line plotting

## 📈 Analysis and Plotting

### Basic Plotting
```python
from fastj_plotting import *

# Load your results
results, metadata = load_fastj_results('my_study')

# Single species profile
fig, ax = plot_species_profile(results, metadata, 'NO2', run_index=0)

# All runs for one species (with statistics)
fig, ax = plot_all_runs_species(results, metadata, 'O3', max_runs=50)

# Compare multiple species
species_list = ['NO2', 'O3', 'H2O2']
fig, ax = plot_species_comparison(results, metadata, species_list)

# Surface values bar chart
fig, ax = plot_surface_values_all_species(results, metadata, top_n=20)
```

### Quick Analysis
```python
# One-liner plots
quick_species_plot('my_study', 'NO2')
quick_all_runs_plot('my_study', 'O3')

# See available species
species = list_available_species('my_study')
```

### Condition Tracking
```python
# Look up run conditions by hash
lookup_run_conditions('69ec6eea3727')

# Find runs with specific conditions
similar_runs = find_runs_by_conditions(
    height_range=(2000, 3000),
    aod_range=(0.1, 0.2)
)
```

## 🚀 Advanced Features

### Checkpointing System
- **Automatic resume**: Interrupted runs continue from last successful case
- **Hash-based tracking**: Each parameter combination gets unique identifier
- **Progress monitoring**: Track completion via `fastj_checkpoint.json`

### Memory Management
- **Dask arrays**: Handle large datasets efficiently
- **Chunked processing**: Process in manageable batches
- **Multiple storage formats**: numpy (compatible), zarr (efficient), HDF5 (compressed)

### Result Storage Options
```python
# Individual timestamped files (default)
append_to_existing = False

# Combined growing file (recommended for studies)
append_to_existing = True
fixed_run_name = 'my_study'
```

## 🔧 Troubleshooting

### Common Issues

#### **"Species not found" errors**
```python
# Check available species
species = list_available_species('your_run_name')
print(species)

# Use exact names (case-sensitive)
plot_species_profile(results, metadata, 'NO2')  # Correct
plot_species_profile(results, metadata, 'no2')  # Wrong
```

#### **"No positive j-values" warnings**
```python
# Use linear scale for problematic species
plot_species_profile(results, metadata, 'NO2', log_scale=False)

# Try different species
plot_species_profile(results, metadata, 'O3')  # Usually has good data
```

#### **Fast-J execution failures**
1. **Check executable permissions**: `chmod +x fastJX`
2. **Verify CTM_atmo.dat exists** in working directory
3. **Test Fast-J manually** with sample input files
4. **Check file paths** in configuration

#### **Memory issues with large runs**
```python
# Process in smaller chunks
max_runs = 100  # Instead of None (all runs)

# Use zarr format for efficiency
pip install zarr
```

#### **File path issues**
- Use **absolute paths** for reliability
- Check that **all CSV files exist** at specified locations
- Ensure **directory permissions** allow writing

### Debug Mode
```python
# Enable debug output in multiRunFastJ.py
debug = 2  # More verbose output

# Check what's being processed
print(f"AOD shape: {aod_data.shape}")
print(f"Height values: {height_data[:5]}")
```

### Getting Help
1. **Check the summary files**: `my_study_summary.txt` contains run details
2. **Verify input data**: Ensure CSV files have correct format
3. **Test with 1 case first**: Set `max_runs = 1` for debugging
4. **Check Fast-J output**: Look at `jVals.dat` and `meanIntensity.dat`

## 📄 Output Files

### Automatic Outputs
- **Results**: `my_study_results.npz` (or .zarr/.h5)
- **Metadata**: `my_study_metadata.pkl` 
- **Summary**: `my_study_summary.txt` (human-readable)
- **Checkpoint**: `fastj_checkpoint.json` (progress tracking)
- **Conditions**: `run_conditions.json` (parameter database)

### Fast-J Outputs (temporary)
- `CTM_GrdCld.dat`: Atmospheric conditions input
- `FJX_scat-rsp.dat`: Aerosol properties input  
- `jVals.dat`: Photolysis rates output
- `meanIntensity.dat`: Mean intensity output

---

## 🎉 You're Ready!

1. **Update the file paths** in `multiRunFastJ.py`
2. **Prepare your CSV data** in the correct format
3. **Test with 1 case**: `max_runs = 1`
4. **Scale up gradually**: 10 → 100 → all cases
5. **Analyze results** with the plotting module

For questions or issues, check the troubleshooting section above!
