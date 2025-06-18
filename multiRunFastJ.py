#!/usr/bin/env python3
import sys
import os
import numpy as np
import csv
import pandas as pd
import subprocess
import hashlib
import dask.array as da
from pathlib import Path
import json
import time


debug = 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def pressure_to_alt(pressure, temp):
    # Compute height from pressure (ht in cm)
    # Conversion factor for pressure to column density
    masfac = 2.1211245975780716E+022
    alts = np.zeros(len(pressure))
    alts[0] = 1600000.0 * np.log10(1013.25 / pressure[0])
    for i in range(0, len(pressure) - 1):
        scale = 1.3806E-19 * masfac * temp[i]
        # below altitudes are in cm
        alts[i + 1] = alts[i] - (np.log(pressure[i + 1] / pressure[i]) * scale)
    return alts


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_hg_moments(g):
    # adding 2l+1 factor
    # only return moments for l=1:7 (not zeroth moment)
    return np.array([
        3.0 * g, 5.0 * g**2, 7.0 * g**3, 9.0 * g**4, 11.0 * g**5, 13.0 * g**6,
        15.0 * g**7
    ])


# =============================================================================
# READING FUNCTIONS
# =============================================================================
def read_aerosol_heights(filepath):
    """
    Read aerosol layer top heights from HSRL CSV file.
    First line is header, remaining lines are height values.

    Parameters:
    -----------
    filepath : str
        Path to the HSRL heights CSV file

    Returns:
    --------
    heights : numpy.ndarray
        Array of aerosol layer top heights
    header : str
        Header string from the file
    """
    heights = []
    header = ""

    with open(filepath, 'r') as file:
        reader = csv.reader(file)

        # Read header (first line)
        header = next(reader)[0]  # Assuming single column header

        # Read height values
        for row in reader:
            if row and row[0].strip():  # Skip empty rows
                try:
                    height = float(row[0])
                    heights.append(height)
                except ValueError:
                    # Skip any non-numeric values
                    continue

    return np.array(heights), header


def read_aerosol_property_data(filepath):
    """
    Read AOD data from CSV file where first row contains wavelengths
    and subsequent rows contain AOD measurements.

    Parameters:
    -----------
    filepath : str
        Path to the AOD CSV file

    Returns:
    --------
    wavelengths : numpy.ndarray
        Array of wavelengths in nm
    aod_measurements : numpy.ndarray
        2D array where each row is an AOD measurement set across all wavelengths
    """
    # Read the CSV file
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # First row contains wavelengths
    wavelengths = np.array([float(wl) for wl in data[0]])

    # Remaining rows contain retrieval values
    aerosol_properties = []
    for row in data[1:]:
        aod_row = [float(val) for val in row if val.strip()]  # Handle empty cells
        aerosol_properties.append(aod_row)

    aerosol_properties = np.array(aerosol_properties)

    return wavelengths, aerosol_properties


# =============================================================================
# HASH AND CHECKPOINT FUNCTIONS
# =============================================================================
def save_run_conditions(run_hash, wavelengths, aod_values, ssa_values, g_values, 
                        height_value, run_index, conditions_file='run_conditions.json'):
    """
    Save the aerosol conditions for a specific run hash.

    Parameters:
    -----------
    run_hash : str
        Unique hash for this run
    wavelengths : array
        Wavelength array
    aod_values : array
        AOD values for this run
    ssa_values : array
        SSA values for this run
    g_values : array
        G parameter values for this run
    height_value : float
        Aerosol layer height for this run
    run_index : int
        Original index in the input arrays
    conditions_file : str
        JSON file to store conditions
    """
    from datetime import datetime

    # Load existing conditions
    if os.path.exists(conditions_file):
        with open(conditions_file, 'r') as f:
            conditions_db = json.load(f)
    else:
        conditions_db = {}

    # Store conditions for this hash
    conditions_db[run_hash] = {
        'timestamp': datetime.now().isoformat(),
        'run_index': int(run_index),
        'height_m': float(height_value),
        'wavelengths_nm': wavelengths.tolist(),
        'aod_values': aod_values.tolist(),
        'ssa_values': ssa_values.tolist(),
        'g_values': g_values.tolist(),
        'aod_550nm': float(np.interp(550.0, wavelengths, aod_values)),  # AOD at 550nm
        'ssa_550nm': float(np.interp(550.0, wavelengths, ssa_values)),  # SSA at 550nm
        'g_550nm': float(np.interp(550.0, wavelengths, g_values)),      # G at 550nm
        'aod_mean': float(np.mean(aod_values)),
        'ssa_mean': float(np.mean(ssa_values)),
        'g_mean': float(np.mean(g_values))
    }

    # Save back to file
    with open(conditions_file, 'w') as f:
        json.dump(conditions_db, f, indent=2)


def enhanced_generate_run_hash(aod_values, ssa_values, g_values, height_value,
                               wavelengths, run_index):
    """
    Enhanced version that generates hash AND saves conditions.

    This replaces the original generate_run_hash function.
    """
    # Generate hash (same as before)
    param_str = (
        f"{wavelengths.tobytes().hex()}"
        f"{aod_values.tobytes().hex()}"
        f"{ssa_values.tobytes().hex()}"
        f"{g_values.tobytes().hex()}"
        f"{height_value}"
    )

    import hashlib
    hash_obj = hashlib.md5(param_str.encode())
    run_hash = hash_obj.hexdigest()[:12]

    # Save conditions for later lookup
    save_run_conditions(run_hash, wavelengths, aod_values, ssa_values,
                        g_values, height_value, run_index)

    return run_hash


def generate_run_hash(aod_values, ssa_values, g_values, height_value, wavelengths):
    """
    Generate a unique hash for a specific combination of aerosol properties.

    Parameters:
    -----------
    aod_values : array
        AOD values for this run
    ssa_values : array
        SSA values for this run
    g_values : array
        G parameter values for this run
    height_value : float
        Aerosol layer height for this run
    wavelengths : array
        Wavelength array

    Returns:
    --------
    hash_str : str
        Unique hash string for this parameter combination
    """
    # Create a concatenated string of all parameters
    param_str = (
        f"{wavelengths.tobytes().hex()}"
        f"{aod_values.tobytes().hex()}"
        f"{ssa_values.tobytes().hex()}"
        f"{g_values.tobytes().hex()}"
        f"{height_value}"

    )
    # Generate MD5 hash (faster than SHA256 for this purpose)
    hash_obj = hashlib.md5(param_str.encode())
    return hash_obj.hexdigest()[:12]  # Use first 12 characters


def load_checkpoint_file(checkpoint_file='fastj_checkpoint.json'):
    """
    Load completed run hashes from checkpoint file.

    Parameters:
    -----------
    checkpoint_file : str
        Path to checkpoint file

    Returns:
    --------
    completed_hashes : set
        Set of completed run hashes
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            return set(data.get('completed_runs', []))
        except Exception as e:
            print(f"Warning: Could not load checkpoint file: {e}")
            return set()
    return set()


def save_checkpoint(run_hash, checkpoint_file='fastj_checkpoint.json'):
    """
    Save completed run hash to checkpoint file.

    Parameters:
    -----------
    run_hash : str
        Hash of completed run
    checkpoint_file : str
        Path to checkpoint file
    """
    # Load existing data
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
        except:
            data = {'completed_runs': []}
    else:
        data = {'completed_runs': []}

    # Add new hash
    if run_hash not in data['completed_runs']:
        data['completed_runs'].append(run_hash)

    # Save back to file
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


# =============================================================================
# WRITING FUNCTIONS
# =============================================================================
def write_JX_scatFile(wavelengths, aod_mode1, aod_mode2, ssa_mode1, ssa_mode2, g_mode1, g_mode2, aer_id='UP'):
    # Set aerosol effective radius and density (currently not used)
    reff_mode1 = 0.221
    reff_mode2 = 0.386
    rho_mode1 = 1.630
    rho_mode2 = 1.630
    fjxScatFile = 'FJX_scat-rsp.dat'

    # optional aerosol ID values
    # aer_id = 'UP'

    # ==========================================
    # Print FJX_scat-rsp.dat file
    # ==========================================
    with open(fjxScatFile, 'w') as f:
        f.write(
            "(FJX_scat-rsp.dat) Aerosol scatter phase fns " +
            "(from RSP ACTIVATE data)" +
            "\n")
        f.write("%s %s %s %s\n" %
                ("####|__name(a12)_| R-eff  rho| notes: aerosol species 01 = ",
                 aer_id, "; species 02 = ", aer_id))
        f.write("w(nm)  aod  ss-alb  pi(1) pi(2) pi(3) pi(4) pi(5) pi(6) pi(7)" +
                "\n")

        # Mode 1
        f.write("%4s %6.3f %6.3f\n" % ("01", reff_mode1, rho_mode1))
        for i in range(0, len(wavelengths)):
            # Compute the moments using asymmetry parameters
            fine_legendre_moments = get_hg_moments(g_mode1[i])
            # Convert moments to printable string
            leg_str = np.array2string(fine_legendre_moments,
                                      precision=3,
                                      separator=' ',
                                      floatmode='fixed')
            f.write("%5s %6.3f %6.3f %s \n" %
                    (wavelengths[i], aod_mode1[i], ssa_mode1[i], leg_str[1:-1]))

        # Mode 2
        f.write("%4s %6.3f %6.3f\n" % ("02", reff_mode2, rho_mode2))
        for i in range(0, len(wavelengths)):
            # Compute the moments using asymmetry parameters
            coarse_legendre_moments = get_hg_moments(g_mode2[i])
            leg_str = np.array2string(coarse_legendre_moments,
                                      precision=3,
                                      separator=' ',
                                      floatmode='fixed')
            f.write("%5s %6.3f %6.3f %s \n" %
                    (wavelengths[i], aod_mode2[i], ssa_mode2[i], leg_str[1:-1]))
        f.write("%4i\n" % (0))
        f.write(
            "===============================================================\n"
        )


def write_CTM_GrdCld(layer1_ht, layer2_ht=0):
    """
    This function currently assumes 1 singular aerosol layer top height,
    and we don't take into account layer thickness. In the future we likely
    should account for specific layer thickness(es).
    """

    filename_atmo = 'CTM_atmo.dat'

    # ensure layer heights are properly converted to km (they're imported in
    # m)
    if layer1_ht > 50:
        layer1_ht = layer1_ht / 1000.0

    # Read the atmospheric file
    data = np.loadtxt(filename_atmo, skiprows=1)

    # Define variables from control file
    pressure = data[:, 1]
    temp = data[:, 2]
    rh = data[:, 3]

    # Compute height (in cm) from pressure/temp
    alts = pressure_to_alt(pressure, temp)

    # Get aerosols in desired location (Find altitudes closest to aerosl
    # layer height values)
    aer_p = np.zeros([len(pressure), 2])
    aer_nda = np.zeros([len(pressure), 2])

    if debug > 1:
        print(f'aer_p = {aer_p}, aer_p shape = {aer_p.shape}')
        print(f'aer_nda = {aer_nda}, aer_nda shape = {aer_nda.shape}')
        print(f'alts = {alts}')

    # Start with only 1 mode (i.e., aerosol type)
    z1 = find_nearest(alts, layer1_ht * 100000)
    idx1 = alts <= z1
    z2 = find_nearest(alts, layer2_ht * 100000)
    idx2 = alts < z2

    if debug > 1:
        print(f'z1 = {z1}')
        print(f'z2 = {z2}')
        print(f'idx1 = {idx1}')
        print(f'idx2 = {idx2}')
    aer_p[idx1, 0] = 1.0
    aer_p[idx2, 1] = 1.0
    aer_nda[idx1, 0] = 100
    aer_nda[idx2, 1] = 101

    # Other variables needed in CTM file
    header = "Input for stand-alone version of Fast-JX code    " + \
             "T42L60, Cycle 36, Year 2005"
    year = 2022
    doy = 181
    month = 6
    gmt = 15.0
    latitude = 36.87
    longitude = -76.02
    psurf = 1009.6
    albsurf = 0.1
    fg0 = 1.10

    # Write the input file
    CTMfile = 'CTM_GrdCld.dat'
    with open(CTMfile, 'w') as f:
        f.write(header + "\n")
        f.write("%-10s \t %-20s\n" % (str(year), "Year            "))
        f.write("%-10s \t %-20s\n" % (str(doy), "Day of year     "))
        f.write("%-10s \t %-20s\n" % (str(month), "Month           "))
        f.write("%-10s \t %-20s\n" % (str(gmt), "GMT             "))
        f.write("%-10s \t %-20s\n" % (str(latitude), "Latitude        "))
        f.write("%-10s \t %-20s\n" % (str(longitude), "Longitude       "))
        f.write("%-10s \t %-20s\n" % (str(psurf), "Surface pressure"))
        f.write("%-10s \t %-20s\n" % (str(albsurf), "Surface albedo"))
        f.write("%-10s \t %-20s\n" %
                (str(fg0), "FG0 = asymm factor for direct beam equiv tau"))
        f.write(
            "L57   Pressure     Temp     RH       Z(m)  AER-P  NDA  AER-P  NDA" +
            "\n")
        for i in range(0, len(pressure)):
            f.write("%-4i %9.3f %8.1f %6.2f %10.2f %6.1f %4i %6.1f %4i\n" %
                    (i+1, pressure[i], temp[i], rh[i], alts[i]/100,
                     aer_p[i, 0], int(aer_nda[i, 0]), aer_p[i, 1],
                     int(aer_nda[i, 1])))


# =============================================================================
# FAST-J EXECUTION AND OUTPUT READING
# =============================================================================
def run_fastj_single(executable_path='./fastJX', timeout=60):
    """
    Run Fast-J executable once and return success status.

    Parameters:
    -----------
    executable_path : str
        Path to Fast-J executable
    timeout : int
        Timeout in seconds

    Returns:
    --------
    success : bool
        True if execution was successful
    """
    try:
        result = subprocess.run(
            [executable_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except Exception:
        return False


def parse_jvals_file(jvals_file='jVals.dat'):
    """
    Parse Fast-J j-values output file.

    Parameters:
    -----------
    jvals_file : str
        Path to jVals.dat file

    Returns:
    --------
    j_data : dict
        Dictionary containing:
        - 'species': list of chemical species names
        - 'altitudes': array of altitudes (km)
        - 'j_values': 2D array (n_altitudes x n_species)
        - 'level_numbers': array of level numbers
    """
    try:
        with open(jvals_file, 'r') as f:
            lines = f.readlines()

        # Parse header to get species names (line 2, after "Fast-J v7.4")
        header_line = lines[1].strip()
        species_names = header_line.split()[2:]  # Skip "L=", "ZKM"

        # Parse data lines (skip first 2 header lines)
        data_lines = [line.strip() for line in lines[2:] if line.strip()]

        level_numbers = []
        altitudes = []
        j_values = []

        for line in data_lines:
            values = line.split()
            level_numbers.append(int(values[0]))
            altitudes.append(float(values[1]))
            # Convert j-values to float (scientific notation)
            j_row = [float(val) for val in values[2:]]
            j_values.append(j_row)

        return {
            'species': species_names,
            'altitudes': np.array(altitudes),
            'j_values': np.array(j_values),
            'level_numbers': np.array(level_numbers)
        }

    except Exception as e:
        print(f"Error parsing j-values file {jvals_file}: {e}")
        return None


def parse_mean_intensity_file(intensity_file='meanIntensity.dat'):
    """
    Parse Fast-J mean intensity output file.

    Parameters:
    -----------
    intensity_file : str
        Path to meanIntensity.dat file

    Returns:
    --------
    intensity_data : dict
        Dictionary containing:
        - 'wavelengths': array of wavelengths (nm)
        - 'altitudes': array of altitudes (km)
        - 'intensities': 2D array (n_altitudes x n_wavelengths)
        - 'metadata': dictionary with solar conditions
    """
    try:
        with open(intensity_file, 'r') as f:
            lines = f.readlines()

        # Parse metadata from summary line (line 2)
        summary_line = lines[1].strip()
        # Extract values like albedo, SZA, etc.
        summary_parts = summary_line.split()
        metadata = {
            'albedo': float(summary_parts[1]),
            'sza': float(summary_parts[2]),
            'u0': float(summary_parts[3]),
            'f_incident': float(summary_parts[4]),
            'f_reflected': float(summary_parts[5]),
            'f_solar': float(summary_parts[6])
        }

        # Parse wavelength header (line 4)
        wl_line = lines[3].strip()
        wavelengths = [float(wl) for wl in wl_line.split()[1:]]  # Skip "wvl:"

        # Parse data lines (starting from line 6, skip header lines)
        data_lines = [line.strip() for line in lines[5:] if line.strip()]

        altitudes = []
        intensities = []

        for line in data_lines:
            values = line.split()
            altitudes.append(float(values[1]))  # Second column is altitude
            # Mean intensities for each wavelength
            intensity_row = [float(val) for val in values[2:]]
            intensities.append(intensity_row)

        return {
            'wavelengths': np.array(wavelengths),
            'altitudes': np.array(altitudes),
            'intensities': np.array(intensities),
            'metadata': metadata
        }

    except Exception as e:
        print(f"Error parsing mean intensity file {intensity_file}: {e}")
        return None


def read_fastj_output(store_mean_intensity=False, jvals_file='jVals.dat',
                      intensity_file='meanIntensity.dat'):
    """
    Read Fast-J output files and extract results.

    Parameters:
    -----------
    store_mean_intensity : bool
        Whether to also read and store mean intensity data
    jvals_file : str
        Path to j-values output file
    intensity_file : str
        Path to mean intensity output file

    Returns:
    --------
    output_data : dict
        Dictionary containing parsed Fast-J output data
    """
    output_data = {}

    # Always read j-values (primary output)
    j_data = parse_jvals_file(jvals_file)
    if j_data is not None:
        output_data['j_values'] = j_data
    else:
        print(f"Failed to read j-values from {jvals_file}")
        return None

    # Optionally read mean intensity
    if store_mean_intensity:
        intensity_data = parse_mean_intensity_file(intensity_file)
        if intensity_data is not None:
            output_data['mean_intensity'] = intensity_data
        else:
            print(f"Warning: Failed to read mean intensity from {intensity_file}")

    return output_data


def extract_output_arrays(output_data, flatten_method='altitude_integrated'):
    """
    Extract numerical arrays from parsed output for efficient storage.
    Parameters:
    -----------
    output_data : dict
        Parsed Fast-J output data
    flatten_method : str
        How to flatten 2D j-values array:
        - 'flatten': Simple flatten to 1D
        - 'altitude_integrated': Integrate over altitude
        - 'surface_only': Only surface values
        - 'full_2d': Keep as 2D array
    Returns:

    --------
    result_array : numpy.ndarray
        Flattened/processed array for storage
    metadata : dict
        Metadata about the processing
    """
    if output_data is None or 'j_values' not in output_data:
        return np.array([]), {}

    j_values = output_data['j_values']['j_values']  # 2D array
    altitudes = output_data['j_values']['altitudes']
    species = output_data['j_values']['species']

    metadata = {
        'species': species,
        'altitudes': altitudes,
        'flatten_method': flatten_method,
        'n_species': len(species),
        'n_altitudes': len(altitudes)
    }

    if flatten_method == 'flatten':
        result_array = j_values.flatten()

    elif flatten_method == 'altitude_integrated':
        # Integrate j-values over altitude (weighted by layer thickness)
        # Assuming uniform layer spacing for now
        dz = np.diff(altitudes)
        if len(dz) > 0:
            # Use average layer thickness
            avg_dz = np.mean(dz)
            result_array = np.sum(j_values * avg_dz, axis=0)  # Sum over altitudes
        else:
            result_array = j_values[0, :]  # Single altitude case

    elif flatten_method == 'surface_only':
        # Take surface values (lowest altitude, highest index)
        result_array = j_values[-1, :]  # Last row (lowest altitude)
        metadata['selected_altitude'] = altitudes[-1]

    elif flatten_method == 'full_2d':
        # Keep full 2D structure
        result_array = j_values

    else:
        raise ValueError(f"Unknown flatten_method: {flatten_method}")

    # Add mean intensity if available
    if 'mean_intensity' in output_data:
        metadata['has_mean_intensity'] = True
        # Could extend to include mean intensity in result_array if needed

    return result_array, metadata


# =============================================================================
# OUTPUT SAVING/LOADING FUNCTIONS
# =============================================================================
def save_results(results_array, metadata, output_dir='fastj_results',
                 run_name=None, save_format='numpy', append_mode=False):
    """
    Save Fast-J results and metadata to disk for later analysis.

    Parameters:
    -----------
    results_array : dask.array
        Dask array containing Fast-J results
    metadata : dict
        Metadata about the runs
    output_dir : str
        Directory to save results
    run_name : str, optional
        Name for this run set (default: timestamp)
    save_format : str
        Format to save results ('zarr', 'numpy', 'hdf5')
    append_mode : bool
        If True, append to existing file; if False, create new timestamped file

    Returns:
    --------
    saved_files : dict
        Dictionary of saved file paths
    """
    import datetime
    import pickle
    import os
    import numpy as np
    import dask.array as da

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Handle run naming based on append mode
    if append_mode and run_name is None:
        # Try to find existing file to append to
        existing_files = []
        for ext in ['.npz', '.zarr', '.h5']:
            pattern_files = [f for f in os.listdir(output_dir) if f.endswith(f'_results{ext}')]
            existing_files.extend(pattern_files)

        if existing_files:
            # Use the most recent existing file
            existing_files.sort()
            latest_file = existing_files[-1]
            if '_results.' in latest_file:
                run_name = latest_file.split('_results.')[0]
                print(f"Append mode: Using existing run name '{run_name}'")
            else:
                # Fallback to timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"fastj_run_{timestamp}"
        else:
            # No existing files, create new
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"fastj_run_{timestamp}"
            print(f"Append mode: No existing files found, creating new run '{run_name}'")

    # Generate run name if not provided and not in append mode
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"fastj_run_{timestamp}"

    print(f"Saving results to: {output_dir}")
    print(f"Run name: {run_name}")
    print(f"Save format: {save_format}")
    print(f"Append mode: {append_mode}")

    # Try different save formats in order of preference
    results_file = None

    if save_format == 'zarr':
        try:
            import zarr
            results_file = os.path.join(output_dir, f"{run_name}_results.zarr")

            if append_mode and os.path.exists(results_file):
                # Load existing data and append
                existing_array = da.from_zarr(results_file)
                print(f"  Found existing zarr with shape {existing_array.shape}")

                # Combine arrays
                combined_array = da.concatenate([existing_array, results_array], axis=0)
                combined_array.to_zarr(results_file, overwrite=True)
                print(f"  ✓ Appended to zarr: {results_file} (new shape: {combined_array.shape})")
            else:
                # Create new file
                results_array.to_zarr(results_file, overwrite=True)
                print(f"  ✓ Results saved to zarr: {results_file}")

        except ImportError:
            print("  ! zarr not available, falling back to numpy format")
            save_format = 'numpy'
        except Exception as e:
            print(f"  ! zarr save failed ({e}), falling back to numpy format")
            save_format = 'numpy'

    if save_format == 'numpy':
        try:
            results_file = os.path.join(output_dir, f"{run_name}_results.npz")

            if append_mode and os.path.exists(results_file):
                # Load existing data and append
                existing_data = np.load(results_file)
                existing_results = existing_data['results']
                existing_hashes = existing_data.get('run_hashes', [])

                print(f"  Found existing numpy file with shape {existing_results.shape}")

                # Combine arrays
                new_results = results_array.compute()
                combined_results = np.concatenate([existing_results, new_results], axis=0)
                combined_hashes = list(existing_hashes) + metadata.get('run_hashes', [])

                np.savez_compressed(results_file,
                                    results=combined_results,
                                    run_hashes=combined_hashes)
                print(f"  ✓ Appended to numpy: {results_file} (new shape: {combined_results.shape})")
            else:
                # Create new file
                results_np = results_array.compute()
                np.savez_compressed(results_file,
                                    results=results_np,
                                    run_hashes=metadata.get('run_hashes', []))
                print(f"  ✓ Results saved to numpy: {results_file}")

        except Exception as e:
            print(f"  ✗ numpy save failed: {e}")
            return None

    elif save_format == 'hdf5':
        try:
            import h5py
            results_file = os.path.join(output_dir, f"{run_name}_results.h5")

            if append_mode and os.path.exists(results_file):
                # Load existing data and append
                with h5py.File(results_file, 'r') as f:
                    existing_results = f['results'][:]
                    existing_hashes = [h.decode() for h in f.get('run_hashes', [])]

                print(f"  Found existing HDF5 with shape {existing_results.shape}")

                # Combine arrays
                new_results = results_array.compute()
                combined_results = np.concatenate([existing_results, new_results], axis=0)
                combined_hashes = existing_hashes + metadata.get('run_hashes', [])

                with h5py.File(results_file, 'w') as f:
                    f.create_dataset('results', data=combined_results, compression='gzip')
                    f.create_dataset('run_hashes', data=[h.encode() for h in combined_hashes])
                print(f"  ✓ Appended to HDF5: {results_file} (new shape: {combined_results.shape})")
            else:
                # Create new file
                results_np = results_array.compute()
                with h5py.File(results_file, 'w') as f:
                    f.create_dataset('results', data=results_np, compression='gzip')
                    f.create_dataset('run_hashes', data=[h.encode() for h in metadata.get('run_hashes', [])])
                print(f"  ✓ Results saved to HDF5: {results_file}")

        except ImportError:
            print("  ! h5py not available, falling back to numpy format")
            return save_results(results_array, metadata, output_dir, run_name, 'numpy', append_mode)
        except Exception as e:
            print(f"  ! HDF5 save failed ({e}), falling back to numpy format")
            return save_results(results_array, metadata, output_dir, run_name, 'numpy', append_mode)

    if results_file is None:
        print("  ✗ All save formats failed")
        return None

    # Save/update metadata (always update in append mode)
    metadata_file = os.path.join(output_dir, f"{run_name}_metadata.pkl")

    if append_mode and os.path.exists(metadata_file):
        # Load existing metadata and merge
        try:
            with open(metadata_file, 'rb') as f:
                existing_metadata = pickle.load(f)

            # Merge run hashes
            existing_hashes = set(existing_metadata.get('run_hashes', []))
            new_hashes = set(metadata.get('run_hashes', []))
            combined_hashes = list(existing_hashes.union(new_hashes))

            # Update metadata
            updated_metadata = existing_metadata.copy()
            updated_metadata['run_hashes'] = combined_hashes
            updated_metadata['total_runs'] = len(combined_hashes)
            updated_metadata['last_updated'] = datetime.datetime.now().isoformat()

            # Save updated metadata
            with open(metadata_file, 'wb') as f:
                pickle.dump(updated_metadata, f)
            print(f"  ✓ Metadata updated: {metadata_file}")

        except Exception as e:
            print(f"  ! Could not merge metadata ({e}), saving new metadata")
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
    else:
        # Save new metadata
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"  ✓ Metadata saved to: {metadata_file}")
        except Exception as e:
            print(f"  ✗ Failed to save metadata: {e}")

    # Create/update human-readable summary
    summary_file = os.path.join(output_dir, f"{run_name}_summary.txt")
    try:
        # Get current total runs (from file if appending)
        if append_mode and save_format == 'numpy' and os.path.exists(results_file):
            current_data = np.load(results_file)
            total_runs_in_file = current_data['results'].shape[0]
        else:
            total_runs_in_file = metadata.get('total_runs', 'unknown')

        with open(summary_file, 'w') as f:
            f.write("Fast-J Results Summary\n")
            f.write("=====================\n\n")
            f.write(f"Run name: {run_name}\n")
            f.write(f"Last updated: {datetime.datetime.now()}\n")
            f.write(f"Save format: {save_format}\n")
            f.write(f"Append mode: {append_mode}\n")
            f.write(f"Results shape: {results_array.shape} (this session)\n")
            f.write(f"Total runs in file: {total_runs_in_file}\n")
            f.write(f"Output method: {metadata.get('output_flatten_method', 'unknown')}\n")
            f.write(f"Store mean intensity: {metadata.get('store_mean_intensity', 'unknown')}\n")
            f.write(f"Wavelengths: {len(metadata.get('wavelengths', []))}\n")

            # List run hashes from this session
            if metadata.get('run_hashes'):
                f.write("\nNew run hashes from this session:\n")
                for i, hash_val in enumerate(metadata['run_hashes']):
                    f.write(f"  {i+1:3d}: {hash_val}\n")

            if metadata.get('output_metadata'):
                output_meta = metadata['output_metadata']
                f.write(f"\nNumber of species: {output_meta.get('n_species', 'unknown')}\n")
                f.write(f"Number of altitudes: {output_meta.get('n_altitudes', 'unknown')}\n")

                if 'species' in output_meta:
                    f.write("\nChemical species:\n")
                    species = output_meta['species']
                    for i, species_name in enumerate(species):
                        f.write(f"  {i:2d}: {species_name}\n")

        print(f"  ✓ Summary saved to: {summary_file}")
    except Exception as e:
        print(f"  ✗ Failed to save summary: {e}")

    return {
        'results_file': results_file,
        'metadata_file': metadata_file,
        'summary_file': summary_file,
        'run_name': run_name,
        'save_format': save_format,
        'append_mode': append_mode
    }


def load_results(results_file=None, metadata_file=None, output_dir='fastj_results',
                 run_name=None, auto_detect_format=True):
    """
    Load previously saved Fast-J results.

    Parameters:
    -----------
    results_file : str, optional
        Direct path to results file
    metadata_file : str, optional
        Direct path to metadata file
    output_dir : str
        Directory containing saved results
    run_name : str, optional
        Name of run to load (will construct file paths)
    auto_detect_format : bool
        Automatically detect file format

    Returns:
    --------
    results_array : numpy.ndarray or dask.array
        Loaded results array
    metadata : dict
        Loaded metadata
    """
    import pickle

    # Auto-detect files if run_name provided
    if results_file is None and run_name is not None:
        # Try different formats
        for ext in ['.npz', '.zarr', '.h5']:
            candidate = os.path.join(output_dir, f"{run_name}_results{ext}")
            if os.path.exists(candidate):
                results_file = candidate
                break

    if metadata_file is None and run_name is not None:
        metadata_file = os.path.join(output_dir, f"{run_name}_metadata.pkl")

    if results_file is None or metadata_file is None:
        raise ValueError("Must provide either file paths or run_name")

    print(f"Loading results from: {results_file}")

    # Load based on file extension
    try:
        if results_file.endswith('.zarr'):
            import zarr
            results_array = da.from_zarr(results_file)

        elif results_file.endswith('.npz'):
            data = np.load(results_file)
            results_array = data['results']

        elif results_file.endswith('.h5'):
            import h5py
            with h5py.File(results_file, 'r') as f:
                results_array = f['results'][:]

        else:
            raise ValueError(f"Unknown file format: {results_file}")

        print(f"  ✓ Results loaded: shape {results_array.shape}")

    except Exception as e:
        print(f"  ✗ Failed to load results: {e}")
        raise

    # Load metadata
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        print("  ✓ Metadata loaded")
    except Exception as e:
        print(f"  ✗ Failed to load metadata: {e}")
        metadata = {}

    return results_array, metadata


def debug_saved_files(output_dir='fastj_results'):
    """
    Debug function to show exactly what files exist.
    """
    print(f"Debugging saved files in: {output_dir}")
    print("="*50)

    if not os.path.exists(output_dir):
        print(f"❌ Directory {output_dir} does not exist!")
        return

    print(f"✓ Directory exists: {output_dir}")

    # List all files
    all_files = os.listdir(output_dir)
    print(f"All files in directory ({len(all_files)} total):")
    for f in sorted(all_files):
        full_path = os.path.join(output_dir, f)
        size = os.path.getsize(full_path)
        print(f"  - {f} ({size:,} bytes)")

    # Find result files
    result_files = [f for f in all_files if 'results' in f]
    print(f"\nResult files found ({len(result_files)}):")
    for f in result_files:
        print(f"  - {f}")
    
    # Find metadata files  
    metadata_files = [f for f in all_files if 'metadata' in f]
    print(f"\nMetadata files found ({len(metadata_files)}):")
    for f in metadata_files:
        print(f"  - {f}")
    
    # Extract run names
    run_names = set()
    for f in all_files:
        if '_results.' in f:
            run_name = f.split('_results.')[0]
            run_names.add(run_name)
        elif '_metadata.' in f:
            run_name = f.split('_metadata.')[0]
            run_names.add(run_name)
    
    print(f"\nDetected run names ({len(run_names)}):")
    for run_name in sorted(run_names):
        print(f"  - {run_name}")
        
        # Check if both files exist for this run
        has_results = any(f.startswith(f"{run_name}_results.") for f in all_files)
        has_metadata = any(f.startswith(f"{run_name}_metadata.") for f in all_files) 
        
        status = "✓ Complete" if (has_results and has_metadata) else "⚠ Incomplete"
        print(f"    {status} (results: {has_results}, metadata: {has_metadata})")
    
    return list(run_names)


def list_saved_runs(output_dir='fastj_results'):
    """
    List all saved Fast-J runs in the output directory.

    Parameters:
    -----------
    output_dir : str
        Directory to search for saved runs

    Returns:
    --------
    runs : list
        List of available run names with their info
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return []

    # Find all result files
    result_files = []
    for ext in ['.zarr', '.npz', '.h5']:
        pattern = f"*_results{ext}"
        matches = [f for f in os.listdir(output_dir) if f.endswith(f"_results{ext}")]
        result_files.extend(matches)

    # Extract run names
    runs = []
    for f in result_files:
        if f.endswith('_results.zarr'):
            run_name = f.replace('_results.zarr', '')
        elif f.endswith('_results.npz'):
            run_name = f.replace('_results.npz', '')
        elif f.endswith('_results.h5'):
            run_name = f.replace('_results.h5', '')
        else:
            continue
        runs.append(run_name)

    print(f"Found {len(runs)} saved runs in {output_dir}:")
    for run in runs:
        summary_file = os.path.join(output_dir, f"{run}_summary.txt")
        if os.path.exists(summary_file):
            # Read key info from summary
            try:
                with open(summary_file, 'r') as f:
                    lines = f.readlines()
                    info_lines = []
                    for line in lines:
                        if any(key in line for key in ['Results shape:', 'Total runs:', 'Save format:', 'Completed run hashes:']):
                            info_lines.append(line.strip())
                    print(f"  {run}: {' | '.join(info_lines)}")
            except:
                print(f"  {run}: (summary file error)")
        else:
            print(f"  {run}: (no summary)")

    return runs


def update_checkpoint_with_saved_status(checkpoint_file='fastj_checkpoint.json', 
                                        saved_hashes=None):
    """
    Update checkpoint file to track which runs have been saved to disk.

    Parameters:
    -----------
    checkpoint_file : str
        Path to checkpoint file
    saved_hashes : list, optional
        List of hashes that have been saved
    """
    # Load existing checkpoint
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
        except:
            data = {'completed_runs': [], 'saved_runs': []}
    else:
        data = {'completed_runs': [], 'saved_runs': []}

    # Ensure saved_runs key exists
    if 'saved_runs' not in data:
        data['saved_runs'] = []

    # Add saved hashes
    if saved_hashes:
        for hash_val in saved_hashes:
            if hash_val not in data['saved_runs']:
                data['saved_runs'].append(hash_val)

    # Save updated checkpoint
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update checkpoint with saved status: {e}")


def check_installation_requirements():
    """
    Check if optional dependencies are available and suggest installation.

    Returns:
    --------
    available : dict
        Dictionary of available packages
    """
    packages = {
        'zarr': False,
        'h5py': False,
        'dask': True,  # Already imported
        'numpy': True  # Already imported
    }

    try:
        import zarr
        packages['zarr'] = True
    except ImportError:
        pass

    try:
        import h5py
        packages['h5py'] = True
    except ImportError:
        pass

    print("Package availability:")
    print(f"  numpy: {'✓' if packages['numpy'] else '✗'} (required)")
    print(f"  dask:  {'✓' if packages['dask'] else '✗'} (required)")
    print(f"  zarr:  {'✓' if packages['zarr'] else '✗'} (optional, for efficient large arrays)")
    print(f"  h5py:  {'✓' if packages['h5py'] else '✗'} (optional, for HDF5 format)")

    if not packages['zarr'] and not packages['h5py']:
        print("\nRecommendation:")
        print("  Install zarr for efficient large array storage:")
        print("    pip install zarr")
        print("  Or install h5py for HDF5 format:")
        print("    pip install h5py")
        print("  (numpy format will be used as fallback)")

    return packages


def analyze_results_quick(results_array, metadata):
    """
    Quick analysis of Fast-J results.

    Parameters:
    -----------
    results_array : dask.array
        Loaded results array
    metadata : dict
        Results metadata

    Returns:
    --------
    analysis : dict
        Basic analysis results
    """
    print("\nQuick Results Analysis")
    print("=====================")

    # Basic info
    print(f"Array shape: {results_array.shape}")
    print(f"Array size: {results_array.size:,} elements")

    # Get some sample data (compute a small portion)
    if results_array.size > 0:
        if results_array.ndim == 2:
            sample = results_array[:min(5, results_array.shape[0]), :min(10, results_array.shape[1])].compute()
        elif results_array.ndim == 3:
            sample = results_array[:min(2, results_array.shape[0]), :5, :5].compute()
        else:
            sample = results_array.flatten()[:100].compute()

        print(f"Value range: {sample.min():.2e} to {sample.max():.2e}")
        print(f"Sample values: {sample.flatten()[:5]}")

        # Species info if available
        if metadata.get('output_metadata', {}).get('species'):
            species = metadata['output_metadata']['species']
            print(f"Number of species: {len(species)}")
            print(f"First 5 species: {species[:5]}")

    return {
        'shape': results_array.shape,
        'size': results_array.size,
        'metadata': metadata
    }


# =============================================================================
# MULTI-RUN MANAGER
# =============================================================================
def run_multi_fastj_cases(wavelengths, aod_data, ssa_data, g_data, height_data,
                          max_runs=None, checkpoint_file='fastj_checkpoint.json',
                          executable_path='./fastJX', store_mean_intensity=False,
                          output_flatten_method='surface_only',
                          incremental_save=True, save_every=20,
                          output_dir='fastj_results', run_name=None, save_format='numpy'):
    """
    Run Fast-J for multiple aerosol property combinations with checkpointing.

    Parameters:
    -----------
    wavelengths : array
        Wavelength array
    aod_data : array
        2D array of AOD measurements
    ssa_data : array
        2D array of SSA measurements
    g_data : array
        2D array of G parameter measurements
    height_data : array
        1D array of height measurements
    max_runs : int, optional
        Maximum number of runs to execute
    checkpoint_file : str
        Path to checkpoint file
    executable_path : str
        Path to Fast-J executable
    store_mean_intensity : bool
        Whether to read and store mean intensity data
    output_flatten_method : str
        How to process j-values output ('surface_only', 'altitude_integrated', 'flatten', 'full_2d')
    incremental_save : bool
        Whether to save results incrementally during processing
    save_every : int
        Save results every N successful runs (only if incremental_save=True)
    output_dir : str
        Directory to save results
    run_name : str, optional
        Fixed name for result files
    save_format : str
        Format for saving results

    Returns:
    --------
    results_array : dask.array
        Dask array containing all Fast-J results
    run_metadata : dict
        Metadata about the runs including hashes and species info
    """

    # Load completed runs from checkpoint
    completed_hashes = load_checkpoint_file(checkpoint_file)
    print(f"Found {len(completed_hashes)} completed runs in checkpoint file")

    # Determine total number of combinations
    n_aod = len(aod_data)
    n_ssa = len(ssa_data)
    n_g = len(g_data)
    n_heights = len(height_data)

    # For this implementation, assume all property arrays have same length
    # and we pair them index-wise (could be modified for full factorial)
    total_combinations = min(n_aod, n_ssa, n_g, n_heights)

    if max_runs:
        total_combinations = min(total_combinations, max_runs)

    print(f"Total combinations to process: {total_combinations}")
    print(f"Store mean intensity: {store_mean_intensity}")
    print(f"Output processing method: {output_flatten_method}")
    if incremental_save:
        print(f"Incremental saving enabled: save every {save_every} runs")

    # Initialize result storage
    results_list = []
    run_hashes = []
    skipped_count = 0
    output_metadata = None
    runs_since_last_save = 0

    # Process each combination
    for i in range(total_combinations):
        # Get current parameter set
        current_aod = aod_data[i]
        current_ssa = ssa_data[i]
        current_g = g_data[i]
        current_height = height_data[i]

        # Generate hash for this combination
        run_hash = enhanced_generate_run_hash(current_aod, current_ssa, current_g,
                                              current_height, wavelengths, i)

        # Check if already completed
        if run_hash in completed_hashes:
            print(f"Run {i+1}/{total_combinations} (hash: {run_hash}) already completed - skipping")
            skipped_count += 1
            continue

        print(f"Processing run {i+1}/{total_combinations} (hash: {run_hash})")

        # Create input files
        try:
            write_JX_scatFile(wavelengths, current_aod, current_aod,
                              current_ssa, current_ssa, current_g, current_g)
            write_CTM_GrdCld(current_height)

            # Run Fast-J
            success = run_fastj_single(executable_path)

            if success:
                # Read output files
                output_data = read_fastj_output(
                    store_mean_intensity=store_mean_intensity,
                    jvals_file='jVals.dat',
                    intensity_file='meanIntensity.dat'
                )

                if output_data is not None:
                    # Extract arrays for storage
                    result_array, metadata = extract_output_arrays(
                        output_data, flatten_method=output_flatten_method
                    )

                    if len(result_array) > 0:
                        results_list.append(result_array)
                        run_hashes.append(run_hash)
                        runs_since_last_save += 1

                        # Store metadata from first successful run
                        if output_metadata is None:
                            output_metadata = metadata

                        # Save checkpoint AFTER successful processing
                        save_checkpoint(run_hash, checkpoint_file)

                        if debug > 0:
                            print(f"  ✓ Run {run_hash} completed successfully")
                            print(f"    Result shape: {result_array.shape}")

                        # INCREMENTAL SAVE: Save results periodically
                        if incremental_save and runs_since_last_save >= save_every:
                            print(f"\n--- Incremental Save ({len(results_list)} results) ---")

                            # Create temporary results array and metadata
                            temp_results_np = np.array(results_list)
                            if temp_results_np.ndim == 3:
                                chunks = (min(100, len(results_list)), -1, -1)
                            else:
                                chunks = (min(100, len(results_list)), -1)
                            temp_results = da.from_array(temp_results_np, chunks=chunks)

                            temp_metadata = {
                                'run_hashes': run_hashes.copy(),
                                'total_runs': len(results_list),
                                'wavelengths': wavelengths,
                                'checkpoint_file': checkpoint_file,
                                'store_mean_intensity': store_mean_intensity,
                                'output_flatten_method': output_flatten_method,
                                'output_metadata': output_metadata,
                                'incremental_save': True,
                                'last_save_run_count': len(results_list)
                            }

                            # Save incrementally with append mode
                            try:
                                saved_files = save_results(
                                    temp_results, temp_metadata,
                                    output_dir=output_dir,
                                    run_name=run_name,
                                    save_format=save_format,
                                    append_mode=True
                                )

                                if saved_files:
                                    print("  ✓ Incremental save successful")
                                    runs_since_last_save = 0  # Reset counter

                                    # Update checkpoint to track saved runs
                                    update_checkpoint_with_saved_status(
                                        checkpoint_file, run_hashes
                                    )
                                else:
                                    print("  ✗ Incremental save failed")

                            except Exception as e:
                                print(f"  ✗ Incremental save error: {e}")
                                print("  Continuing processing...")

                            print("--- Incremental Save Complete ---\n")
                    else:
                        print(f"  ✗ Run {run_hash} failed - no valid output data")
                else:
                    print(f"  ✗ Run {run_hash} failed - could not parse output")
            else:
                print(f"  ✗ Run {run_hash} failed - Fast-J execution error")

        except Exception as e:
            print(f"  ✗ Run {run_hash} failed with exception: {e}")

    print(f"\nCompleted {len(results_list)} new runs")
    print(f"Skipped {skipped_count} already completed runs")

    # Convert to dask array for memory efficiency
    if results_list:
        # Handle different array shapes based on output method
        results_np = np.array(results_list)
        print(f"Results numpy array shape: {results_np.shape}")

        # Set appropriate chunking based on array dimensions
        if results_np.ndim == 2:
            # 2D array: (n_runs, n_features) - for flattened outputs
            chunks = (min(100, results_np.shape[0]), -1)
        elif results_np.ndim == 3:
            # 3D array: (n_runs, n_altitudes, n_species) - for full_2d outputs
            chunks = (min(100, results_np.shape[0]), -1, -1)
        else:
            # Handle other dimensions
            chunks = tuple(min(100, dim) if i == 0 else -1 for i, dim in enumerate(results_np.shape))

        results_array = da.from_array(results_np, chunks=chunks)
        print(f"Results dask array shape: {results_array.shape}")
        print(f"Dask chunks: {chunks}")

        if output_metadata:
            print(f"Number of species: {output_metadata.get('n_species', 'unknown')}")
            print(f"Species: {output_metadata.get('species', [])[:5]}...")  # Show first 5
            if output_flatten_method == 'full_2d':
                print(f"Number of altitudes: {output_metadata.get('n_altitudes', 'unknown')}")
    else:
        results_array = da.empty((0, 0))

    # Prepare comprehensive metadata
    run_metadata = {
        'run_hashes': run_hashes,
        'total_runs': len(results_list),
        'wavelengths': wavelengths,
        'checkpoint_file': checkpoint_file,
        'store_mean_intensity': store_mean_intensity,
        'output_flatten_method': output_flatten_method,
        'output_metadata': output_metadata,
        'incremental_save_used': incremental_save,
        'save_every': save_every if incremental_save else None
    }

    return results_array, run_metadata


# currently not used
def check_fastj_executable(executable_path='./fastJX'):
    """
    Check if Fast-J executable exists and is executable.

    Parameters:
    -----------
    executable_path : str
        Path to the Fast-J executable

    Returns:
    --------
    bool : True if executable is found and ready
    """
    if not os.path.exists(executable_path):
        print(f"Error: Fast-J executable not found at {executable_path}")
        return False

    if not os.access(executable_path, os.X_OK):
        print(f"Error: Fast-J executable at {executable_path} is not executable")
        print("Try: chmod +x fastJX")
        return False

    print(f"Fast-J executable found at {executable_path}")
    return True


# Currently not used
def check_required_input_files():
    """
    Check if all required Fast-J input files exist.

    Returns:
    --------
    bool : True if all required files exist
    """
    required_files = [
        'CTM_GrdCld.dat',
        'FJX_scat-rsp.dat',
        'FJX_spec.dat',
        'atmos_std.dat',
        'atmos_h2och4.dat',
        'FJX_scat-aer.dat',
        'FJX_scat-cld.dat',
        'FJX_scat-ssa.dat',
        'FJX_scat-UMa.dat',
        'FJX_j2j.dat'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing required input files: {missing_files}")
        return False

    print("All required input files found")
    return True


# =============================================================================
# MAIN PROGRAM
# =============================================================================
def main():
    print("="*60)
    print("Multi-Run Fast-J with Checkpointing")
    print("="*60)

    # Define directories containing csv files
    csvDir_aerosolProps = '/Users/adambell/Research/photochemistry/fastj-multi-run/aerosol_fastJcsvFiles'
    csvDir_aerosolHeight = '/Users/adambell/Research/photochemistry/fastj-multi-run/aerosol_heights'

    # Define individual property file names
    csvName_AOD = 'AOD_Urban_Pollution.csv'
    csvName_SSA = 'SSA_Urban_Pollution.csv'
    csvName_G = 'G_Urban_Pollution.csv'
    csvName_height = 'HSRL_Urban_Pollution_hts.csv'

    # Get full file paths
    file_name_AOD = os.path.join(csvDir_aerosolProps, csvName_AOD)
    file_name_SSA = os.path.join(csvDir_aerosolProps, csvName_SSA)
    file_name_G = os.path.join(csvDir_aerosolProps, csvName_G)
    file_name_height = os.path.join(csvDir_aerosolHeight, csvName_height)

    print("Loading aerosol property data...")

    # Read all aerosol properties and heights
    wavelengths, aod_data = read_aerosol_property_data(file_name_AOD)
    wavelengths, ssa_data = read_aerosol_property_data(file_name_SSA)
    wavelengths, g_data = read_aerosol_property_data(file_name_G)
    height_data, header = read_aerosol_heights(file_name_height)

    print("Loaded data:")
    print(f"  - Wavelengths: {len(wavelengths)}")
    print(f"  - AOD measurements: {aod_data.shape}")
    print(f"  - SSA measurements: {ssa_data.shape}")
    print(f"  - G measurements: {g_data.shape}")
    print(f"  - Height measurements: {len(height_data)}")

    # ========================================================================
    # CONFIGURATION SECTION - Modify these parameters as needed
    # ========================================================================

    # Number of runs to process (None = all available)
    max_runs = 100  # SET TO 1 FOR TESTING, any number for chunks, or None (all)

    # Output processing options
    store_mean_intensity = False  # Set to True if you want mean intensity data
    output_method = 'full_2d'  # Options: 'surface_only', 'altitude_integrated', 'flatten', 'full_2d'

    # File saving options
    append_to_existing = True
    fixed_run_name = 'Urban_Pollution'
    save_format = 'zarr'  # Options: 'zarr', 'numpy', 'hdf5'

    # Incremental saving (recommended for long runs)
    incremental_save = True  # save results periodically
    save_every = 20  # save after every 20 runs

    # File paths
    checkpoint_file = 'fastj_checkpoint.json'
    executable_path = './fastJX'

    print("\nRun Configuration:")
    print(f"  - Max runs to process: {max_runs if max_runs else 'ALL'}")
    print(f"  - Store mean intensity: {store_mean_intensity}")
    print(f"  - Output method: {output_method}")
    print(f"  - Append to existing: {append_to_existing}")
    print(f"  - Fixed run name: {fixed_run_name}")
    print(f"  - Incremental save: {incremental_save} (every {save_every} runs)")
    print(f"  - Checkpoint file: {checkpoint_file}")
    print(f"  - Executable: {executable_path}")

    # ========================================================================

    # Run multi-case Fast-J
    results_array, metadata = run_multi_fastj_cases(
        wavelengths, aod_data, ssa_data, g_data, height_data,
        max_runs=max_runs,
        checkpoint_file=checkpoint_file,
        executable_path=executable_path,
        store_mean_intensity=store_mean_intensity,
        output_flatten_method=output_method,
        incremental_save=incremental_save,
        output_dir='fastj_results',
        run_name=fixed_run_name,
        save_format=save_format
    )

    print("\nFinal results:")
    print(f"  - Results array shape: {results_array.shape}")
    print(f"  - Total completed runs: {metadata['total_runs']}")
    print("  - Results stored in dask array for memory efficiency")

    # Display some results info if available
    if metadata.get('output_metadata'):
        output_meta = metadata['output_metadata']
        print(f"  - Number of species: {output_meta.get('n_species', 'unknown')}")
        print(f"  - Number of altitudes: {output_meta.get('n_altitudes', 'unknown')}")
        if 'species' in output_meta:
            species_list = output_meta['species']
            print(f"  - Species (first 10): {species_list[:10]}")

    # ========================================================================
    # SAVE RESULTS FOR LATER ANALYSIS
    # ========================================================================
    if results_array.shape[0] > 0:  # Only save if we have results
        print("\nSaving results for later analysis...")
        saved_files = save_results(
            results_array,
            metadata,
            output_dir='fastj_results',
            run_name=fixed_run_name,  # Will auto-generate timestamp name
            save_format=save_format,
            append_mode=append_to_existing
        )

        if saved_files:
            print("Results saved successfully!")
            print("To load later, use:")
            print(f"  results, meta = load_results(run_name='{saved_files['run_name']}')")

    # Optional: Save results to file
    print(f"  - Checkpoint file: {metadata['checkpoint_file']}")

    return results_array, metadata


if __name__ == '__main__':
    main()
