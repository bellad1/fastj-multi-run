# Fast-J Multi-Run Atmospheric Photochemistry Model

A comprehensive Python-based system for running multiple Fast-J photolysis rate
calculations with user provided aerosol properties.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)

## Overview

Necessary aerosol properties to run this code are aerosol optical depth (AOD),
single scattering albedo (SSA), asymmetry factor (G), and aerosol layer top
height. NOTE: this code was originally designed to calculate photolsyis rates
using aerosol property retrievals from the NASA 
[ACTIVATE](https://science.larc.nasa.gov/activate/) campaign and the
[MAPP](https://opg.optica.org/ao/fulltext.cfm?uri=ao-57-10-2394&id=383916)
retrieval framework (Stamnes et al., 2018).

This repository contains tools for:
- Processing retrieved aerosol properties from RSP (Research Scanning Polarimeter) and HSRL (High Spectral Resolution Lidar) measurements with the MAPP retrieval framework.
- Running Fast-J photolysis rate calculations with varying aerosol contditions
- Managing large-scale computational campaigns with checkpointing
- Analyzing and storing results efficiently using Dask arrays

## Features

- **Multi-run capabilities**: Process thousands of aerosol scenarios automatically
- **Checkpointing**: Resume interrupted calculations
- **Memory efficiency**: Uses Dask for large dataset handling
- **Multiple output formats**: Zarr, HDF5, NumPy support
- **Incremental saving**: Save results periodically during long runs
- **Data validation**: Hash-based run identification and duplicate prevention

## System Requirements
- Fast-J executable (compiled from Fortran source)
- Python 3.7+
- Sufficient disk space for results (varies by campaign size)

### Core Dependencies
python dependencies:
```
pip install numpy scipy matplotlib pandas dask
```
fast-JX executable:
```bash
ln -snf /Path/To/FastJX/Executable/fastJX ./fastJX
```

### Optional Dependencies (for enhanced functionality)
```
pip install zarr h5py  # For efficient large array storage
```

## Quick Start
### 1. Data Preparation
This code was designed to focus on impacts of different aerosol types.
Therefore, CSV input data files must contain an 'aerosol_tag', such as
'Urban_Pollution', which is proceeded by the aerosol property and an underscore
(AOD_, SSA_, G_).\
\
As an example, for 'Urban_Pollution', ensure your CSV files are organized as:\
```
aerosol_fastJcsvFiles/
├── AOD_Urban_Pollution.csv
├── SSA_Urban_Pollution.csv
└── G_Urban_Pollution.csv

aerosol_heights/
└── HSRL_Urban_Pollution_hts.csv
```
### 2. Basic Usage
```python
from multiRunFastJ import main

### Run with default configuration
results_array, metadata = main()
```

### 3. Custom Configuration
Edit the configuration section in multiRunFastJ.py:
```python
# Number of runs (None = all available)
max_runs = 100

### Output processing
store_mean_intensity = False
output_method = 'surface_only'  # or 'full_2d', 'altitude_integrated'

### File management
append_to_existing = True
save_format = 'zarr'  # or 'numpy', 'hdf5'
```

## File Structure

```
├── multiRunFastJ.py           # Main multi-run script
├── collocationTest_multiFile.py  # Data collocation and processing
├── AOD_Urban_Pollution.csv    # Aerosol optical depth data
├── HSRL_Urban_Pollution_hts.csv  # Aerosol layer heights
├── jVals.dat                  # Example Fast-J output
├── meanIntensity.dat          # Example mean intensity output
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Data Input Format

### Aerosol Properties
**IMPORTANT:** aerosol properties at the retrieved wavelengths had to be
interpolated/extrapolated to Fast-J wavelengths. This introduces uncertainty
that has not been quantified.

CSV files with wavelengths in header row and aerosol property (AOD, SSA, G)
retrieval values in subsequent rows (need a separate csv file for each
property):
```csv
187.0,191.0,193.0,196.0,...
0.123,0.145,0.156,0.167,...
0.098,0.112,0.125,0.134,...
```
### Height Data
Single column CSV with header (NOTE: header contents not important, just ensure
single line):
```csv
HSRL Aerosol Heights for Urban/Pollution
2450.0
1890.0
3120.0
```

## Output Data

### J-values
Photolysis rates for atmospheric species at different altitudes:
- Surface-only: J-values at ground level
- Full 2D: Complete altitude-species matrix
- Altitude-integrated: Column-integrated values

### Storage Formats
- **Zarr**: Recommended for large datasets
- **NumPy**: Simple .npz format
- **HDF5**: Industry standard scientific format

## Configuration Options

### Processing Methods
- `surface_only`: Extract ground-level photolysis rates
- `full_2d`: Preserve complete altitude×species arrays
- `altitude_integrated`: Integrate over atmospheric column
- `flatten`: Simple 1D flattening

### Checkpointing
Automatic resume capability using fastj_checkpoint.json:
```json
{
  "completed_runs": ["hash1", "hash2", ...],
  "saved_runs": ["hash1", "hash2", ...]
}
```

## Performance Optimization

### Memory Management
- Uses Dask for out-of-core computation
- Configurable chunking for large arrays
- Incremental saving to prevent data loss

### Computational Efficiency
- Duplicate run detection via hashing
- Parallel processing capability (future enhancement)
- Optimized I/O for large campaigns

## Example Workflows

### 1. Small Test Run
```python
# Test with 10 cases
max_runs = 10
output_method = 'surface_only'
save_format = 'numpy'
```
### 2. Process entire ACTIVATE Campaign
```python
# Full dataset processing
max_runs = None  # All available
output_method = 'full_2d'
save_format = 'zarr'
incremental_save = True
save_every = 50
```

### 3. Resume Interrupted Run
```python
# Automatically resumes from checkpoint
append_to_existing = True
fixed_run_name = 'Urban_Pollution'
```

## Results Analysis

### Loading Saved Results
```python
from multiRunFastJ import load_results

# Load by run name
results, metadata = load_results(run_name='Urban_Pollution_20241201')

# Quick analysis
from multiRunFastJ import analyze_results_quick
analysis = analyze_results_quick(results, metadata)
```

### Available Tools
- `list_saved_runs()`: Browse available datasets
- `debug_saved_files()`: Troubleshoot file issues
- `analyze_results_quick()`: Basic statistical analysis

## Troubleshooting

### Common Issues
1. **Missing executable**: Ensure fastJX is compiled and executable. Create
   a symbolic link from the fastJX source directory to your working directory
with 
```bash
ln -snf /Path/To/FastJX/Executable/fastJX ./fastJX
```
2. **Memory errors**: Reduce chunk sizes or use incremental saving
3. **File format errors**: Check CSV structure and headers
4. **Checkpoint corruption**: Delete and restart from clean state

# Debug Functions
```python
# Check saved files
from multiRunFastJ import debug_saved_files
debug_saved_files('fastj_results')

## List available runs
from multiRunFastJ import list_saved_runs
runs = list_saved_runs()
```

## Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/new-analysis)
3. Commit changes (git commit -am 'Add new analysis method')
4. Push to branch (git push origin feature/new-analysis)
5. Create Pull Request

## Citation
If you use this code in your research, please cite:
```
[Your paper citation here]
```

## License
Open-source and not licensed 

## Contact

- **Author**: Adam Bell
- **Email**: adamdrakebell@gmail.com
- **Institution**: N/A

## Acknowledgments

- ACTIVATE, RSP and HRSL science teams for measurement data
- Stamnes et al., for MAPP aerosol property retrievals
- Prather et al., for development of Fast-J software
