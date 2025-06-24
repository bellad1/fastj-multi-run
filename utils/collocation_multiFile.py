import h5py
import numpy as np
from collections import Counter
from scipy import interpolate
import matplotlib.pyplot as plt
# import sys
import os
import csv
import sys


'''
This code is meant to produce the csv files of aerosol properties (AOD, SSA, G)
to run fast-J for the entire ACTIVATE campaign.

The overall flow is:
    1) we set a threshold particular percentage of aerosol in the HSRL column.
    Meaning, if threshold = 0.5, then if half of the vertical bins from HSRL
    aerosol typing contain Marine aerosols, then we assume the retrieved aerosol
    properties are "Marine" (NOTE: this threshold hasn't been rigorously tested)
    2) collocate HSRL and RSP data points based on minimum criteria below
    3) collect RSP-MAPP retrieved SSA, AOD, G and RSP and HSRL aerosol layer heights
    4) interpolate/extrapolate these properties to fast-J wavelengths (NOTE:
    some interp/extrap can introduce unphysical negative values. This code accounts
    and corrects for this. If you use a different interp/extrap scheme ensure you
    avoid negative values.)
    5) write csv files

NOTE: the RSP/HSRL retrieval file names are in main(). You must edit them and their
respective paths before running.
'''

# Threshold for percentage of aerosol in vertical column
threshold = 0.50

# Boolean toggle to control behavior
# Set True to return only the maximum % aerosol type that meets threshold
return_max = True

# Minimum criteria for delta values
min_delta_lat = 0.1
min_delta_lon = 0.1
min_delta_time = 5

fastJX_channels = np.array(
                [187.0, 191.0, 193.0, 196.0, 202.0, 208.0, 211.0, 214.0,
                 261.0, 267.0, 277.0, 295.0, 303.0, 310.0, 316.0, 333.0,
                 383.0, 599.0, 973.0, 1267.0, 1448.0, 1767.0, 2039.0, 2309.0,
                 2748.0, 3404.0, 5362.0]
                )


def read_hdf5_variable(file_path, variable_name):
    """
    Reads a specified variable from an HDF5 file.

    Args:
        file_path (str): The path to the HDF5 file.
        variable_name (str): The name of the variable to read.

    Returns:
        np.ndarray: The data from the specified variable.
    """
    with h5py.File(file_path, 'r') as f:
        data = f[variable_name][:]
    return data


def convert_time_to_seconds_utc(time_utc):
    """
    Converts time in hours (UTC) to seconds since midnight.

    Args:
        time_utc (np.ndarray): Time in UTC (hours).

    Returns:
        np.ndarray: Time in seconds since midnight.
    """
    return time_utc * 60 * 60


def find_closest_time_index(time_target, time_array):
    """
    Finds the closest time index in an array to a target time.

    Args:
        time_target (float): The target time to match.
        time_array (np.ndarray): The array of times to search.

    Returns:
        int: The index of the closest time in the array.
    """
    time_differences = np.abs(time_array - time_target)
    closest_index = np.argmin(time_differences)
    return closest_index


def compute_lat_lon_difference(lat1, lon1, lat2, lon2):
    """
    Computes the differences in latitude and longitude between two points.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        tuple: Differences in latitude and longitude.
    """
    lat_diff = lat1 - lat2
    lon_diff = lon1 - lon2
    return lat_diff, lon_diff


def get_non_nan_values(data):
    """
    Extracts non-NaN values and their indices from a given array.

    Args:
        data (np.ndarray): The input array containing NaN values.

    Returns:
        tuple: (non-NaN values, indices of non-NaN values)
    """
    non_nan_indices = np.where(~np.isnan(data))[0]
    non_nan_values = data[non_nan_indices]
    return non_nan_values, non_nan_indices


def count_unique_values_percentage(data):
    """
    Counts unique values in an array and returns a dictionary with percentages.

    Args:
        data (np.ndarray): The input array with values to count.

    Returns:
        dict: A dictionary with unique values as keys and their percentage of
        the total count as values.
    """
    total_count = len(data)  # Total number of elements in the data
    unique_counts = dict(Counter(data))  # Get the count of each unique value
    unique_percentages = {k: v / total_count for k, v in unique_counts.items()}
    return unique_percentages


def interpolate_to_fastJX_simple(output_channels, variable, method='linear'):
    """
    Simple and robust interpolation that handles extrapolation properly
    and prevents negative AOD values.
    """
    # Define the specific wavelengths to interpolate to (FastJX channels)
    fastJX_channels = np.array([
        187.0, 191.0, 193.0, 196.0, 202.0, 208.0, 211.0, 214.0,
        261.0, 267.0, 277.0, 295.0, 303.0, 310.0, 316.0, 333.0,
        383.0, 599.0, 973.0, 1267.0, 1448.0, 1767.0, 2039.0, 2309.0,
        2748.0, 3404.0, 5362.0
    ])

    # Sort the output_channels and variable together
    sorted_indices = np.argsort(output_channels)
    output_channels_sorted = np.array(output_channels)[sorted_indices]
    variable_sorted = np.array(variable)[sorted_indices]

    print(f"Debug - sorted wavelengths: {output_channels_sorted}")
    valid_mask = ~np.isnan(variable_sorted)

    output_channels_clean = output_channels_sorted[valid_mask]
    variable_clean = variable_sorted[valid_mask]

    # Ensure we have enough points
    if len(output_channels_clean) < 2:
        raise ValueError("Need at least 2 valid data points for interpolation")

    min_measured = np.min(output_channels_clean)
    max_measured = np.max(output_channels_clean)

    print(f"Debug - measurement range: {min_measured} to {max_measured} nm")

    # Method 1: Use scipy's interpolation with extrapolation, then fix negatives
    # This is more robust than trying to fit power laws

    # Create interpolation function that allows extrapolation
    if method == 'linear':
        interp_func = interpolate.interp1d(
            output_channels_clean, variable_clean,
            kind='linear', fill_value='extrapolate', bounds_error=False
        )
    else:
        # For cubic or other methods, use linear extrapolation at boundaries
        interp_func = interpolate.interp1d(
            output_channels_clean, variable_clean,
            kind=method, fill_value='extrapolate', bounds_error=False
        )

    # Get interpolated values
    variable_interpolated = interp_func(fastJX_channels)

    print(f"Debug - after basic interpolation, min value: {np.min(variable_interpolated)}")
    print(f"Debug - negative count: {np.sum(variable_interpolated < 0)}")

    # Handle negative values with physically reasonable approaches

    # For short wavelengths (extrapolation below minimum), use power law trend
    short_mask = fastJX_channels < min_measured
    if np.any(short_mask) and np.any(variable_interpolated[short_mask] <= 0):
        # Use the trend from the first few measured points
        # Typically AOD increases toward shorter wavelengths
        wl1, wl2 = output_channels_clean[0], output_channels_clean[1]
        aod1, aod2 = variable_clean[0], variable_clean[1]

        # Calculate power law exponent from first two points
        # log(AOD2/AOD1) = alpha * log(wl1/wl2)
        if aod1 > 0 and aod2 > 0:
            alpha = np.log(aod1/aod2) / np.log(wl2/wl1)
            # Ensure reasonable alpha (typical range 0.5 to 2.0 for aerosols)
            alpha = np.clip(alpha, 0.3, 3.0)

            # Apply power law: AOD = aod1 * (wl/wl1)^(-alpha)
            short_wl = fastJX_channels[short_mask]
            power_law_values = aod1 * (short_wl / wl1) ** (-alpha)
            variable_interpolated[short_mask] = power_law_values

            print(f"Debug - applied power law for short wavelengths with alpha={alpha:.2f}")

    # For long wavelengths (extrapolation beyond maximum), use exponential decay
    long_mask = fastJX_channels > max_measured
    if np.any(long_mask) and np.any(variable_interpolated[long_mask] <= 0):
        # Use exponential decay based on last few points
        n_pts = min(3, len(output_channels_clean))
        wl_end = output_channels_clean[-n_pts:]
        aod_end = variable_clean[-n_pts:]

        # Fit exponential decay: AOD = A * exp(-B * wl)
        if np.all(aod_end > 0):
            log_aod = np.log(aod_end)
            coeffs = np.polyfit(wl_end, log_aod, 1)
            decay_rate = -coeffs[0]  # Should be positive for decay
            log_A = coeffs[1]

            # Ensure reasonable decay rate
            decay_rate = max(decay_rate, 1e-4)

            long_wl = fastJX_channels[long_mask]
            exp_values = np.exp(log_A) * np.exp(-decay_rate * long_wl)
            variable_interpolated[long_mask] = exp_values

            print(f"Debug - applied exponential decay for long wavelengths with rate={decay_rate:.2e}")

    # Final safety: ensure no negative values (absolute minimum constraint)
    negative_mask = variable_interpolated < 0
    if np.any(negative_mask):
        print(f"Debug - setting {np.sum(negative_mask)} remaining negative values to small positive")
        # Set to small positive value based on the minimum positive value in the data
        min_positive = np.min(variable_clean[variable_clean > 0]) if np.any(variable_clean > 0) else 1e-6
        variable_interpolated[negative_mask] = min_positive * 0.1

    print(f"Debug - final result, min value: {np.min(variable_interpolated)}")

    return variable_interpolated


def interpolate_to_fastJX_improved(output_channels, variable, method='linear',
                                   extrapolation_method='power_law', min_value=0.0):
    """
    Improved interpolation/extrapolation that prevents negative AOD values.

    Parameters:
    -----------
    output_channels : array-like
        Original wavelengths of measurements
    variable : array-like
        AOD values at original wavelengths
    method : str
        Interpolation method ('linear', 'cubic', 'quadratic')
    extrapolation_method : str
        Method for extrapolation ('power_law', 'exponential', 'constant', 'linear_bounded')
    min_value : float
        Minimum allowed value (default 0.0 for AOD)

    Returns:
    --------
    variable_interpolated : array
        AOD values at FastJX wavelengths
    """

    # Define the specific wavelengths to interpolate to (FastJX channels)
    fastJX_channels = np.array([
        187.0, 191.0, 193.0, 196.0, 202.0, 208.0, 211.0, 214.0,
        261.0, 267.0, 277.0, 295.0, 303.0, 310.0, 316.0, 333.0,
        383.0, 599.0, 973.0, 1267.0, 1448.0, 1767.0, 2039.0, 2309.0,
        2748.0, 3404.0, 5362.0
    ])

    # Sort the output_channels and variable together
    sorted_indices = np.argsort(output_channels)
    output_channels_sorted = np.array(output_channels)[sorted_indices]
    variable_sorted = np.array(variable)[sorted_indices]

    # Remove any NaN or negative values from input data
    valid_mask = ~np.isnan(variable_sorted) & (variable_sorted >= 0)
    output_channels_clean = output_channels_sorted[valid_mask]
    variable_clean = variable_sorted[valid_mask]

    if len(output_channels_clean) < 2:
        raise ValueError("Need at least 2 valid data points for interpolation")

    # Determine wavelength ranges
    min_measured = np.min(output_channels_clean)
    max_measured = np.max(output_channels_clean)

    # Separate FastJX channels into interpolation and extrapolation regions
    interpolation_mask = (fastJX_channels >= min_measured) & (fastJX_channels <= max_measured)
    short_extrap_mask = fastJX_channels < min_measured
    long_extrap_mask = fastJX_channels > max_measured

    # Initialize output array
    variable_interpolated = np.zeros_like(fastJX_channels)

    # 1. Handle interpolation region (standard interpolation)
    if np.any(interpolation_mask):
        interp_function = interpolate.interp1d(
            output_channels_clean, variable_clean,
            kind=method, bounds_error=False, fill_value=np.nan
        )
        variable_interpolated[interpolation_mask] = interp_function(
            fastJX_channels[interpolation_mask]
        )

    # 2. Handle short wavelength extrapolation
    if np.any(short_extrap_mask):
        if extrapolation_method == 'power_law':
            # Fit power law using first few points: AOD = a * lambda^(-alpha)
            n_points = min(3, len(output_channels_clean))
            wl_fit = output_channels_clean[:n_points]
            aod_fit = variable_clean[:n_points]

            # Fit in log space: log(AOD) = log(a) - alpha * log(lambda)
            log_wl = np.log(wl_fit)
            log_aod = np.log(np.maximum(aod_fit, 1e-10))  # Avoid log(0)

            # Linear fit in log space
            coeffs = np.polyfit(log_wl, log_aod, 1)
            alpha = -coeffs[0]  # Negative slope becomes positive alpha
            log_a = coeffs[1]

            # Apply power law extrapolation
            short_wl = fastJX_channels[short_extrap_mask]
            variable_interpolated[short_extrap_mask] = np.exp(log_a) * (short_wl ** (-alpha))

        elif extrapolation_method == 'linear_bounded':
            # Linear extrapolation but force positive slope at short wavelengths
            # Use first two points
            wl1, wl2 = output_channels_clean[0], output_channels_clean[1]
            aod1, aod2 = variable_clean[0], variable_clean[1]

            # Calculate slope, but limit it to ensure positive extrapolation
            slope = (aod2 - aod1) / (wl2 - wl1)
            slope = max(slope, 0)  # Force non-negative slope

            short_wl = fastJX_channels[short_extrap_mask]
            variable_interpolated[short_extrap_mask] = aod1 + slope * (short_wl - wl1)

    # 3. Handle long wavelength extrapolation (most critical for negative values)
    if np.any(long_extrap_mask):
        if extrapolation_method == 'power_law':
            # Fit power law using last few points
            n_points = min(4, len(output_channels_clean))
            wl_fit = output_channels_clean[-n_points:]
            aod_fit = variable_clean[-n_points:]

            # Fit power law: AOD = a * lambda^(-alpha)
            log_wl = np.log(wl_fit)
            log_aod = np.log(np.maximum(aod_fit, 1e-10))  # Avoid log(0)

            # Linear fit in log space
            coeffs = np.polyfit(log_wl, log_aod, 1)
            alpha = -coeffs[0]  # Negative slope becomes positive alpha
            log_a = coeffs[1]

            # Apply power law extrapolation
            long_wl = fastJX_channels[long_extrap_mask]
            variable_interpolated[long_extrap_mask] = np.exp(log_a) * (long_wl ** (-alpha))

        elif extrapolation_method == 'exponential':
            # Exponential decay: AOD = a * exp(-b * lambda)
            n_points = min(4, len(output_channels_clean))
            wl_fit = output_channels_clean[-n_points:]
            aod_fit = variable_clean[-n_points:]

            # Fit exponential in semi-log space
            log_aod = np.log(np.maximum(aod_fit, 1e-10))
            coeffs = np.polyfit(wl_fit, log_aod, 1)
            b = -coeffs[0]  # Decay rate
            log_a = coeffs[1]

            # Apply exponential extrapolation
            long_wl = fastJX_channels[long_extrap_mask]
            variable_interpolated[long_extrap_mask] = np.exp(log_a) * np.exp(-b * long_wl)

        elif extrapolation_method == 'constant':
            # Use constant value equal to the last measured point
            variable_interpolated[long_extrap_mask] = variable_clean[-1]

    # 4. Apply minimum value constraint (ensure no negative values)
    variable_interpolated = np.maximum(variable_interpolated, min_value)

    return variable_interpolated


def interpolate_to_fastJX(output_channels, variable, method='linear'):
    # Define the specific wavelengths to interpolate to (FastJX channels)
    fastJX_channels = np.array(
            [187.0, 191.0, 193.0, 196.0, 202.0, 208.0, 211.0, 214.0,
             261.0, 267.0, 277.0, 295.0, 303.0, 310.0, 316.0, 333.0,
             383.0, 599.0, 973.0, 1267.0, 1448.0, 1767.0, 2039.0, 2309.0,
             2748.0, 3404.0, 5362.0]
            )

    # Sort the output_channels and variable together
    sorted_indices = np.argsort(output_channels)
    output_channels_sorted = np.array(output_channels)[sorted_indices]
    variable_sorted = np.array(variable)[sorted_indices]

    # Interpolate/extrapolate to the specific FastJX channels
    interp_function = interpolate.interp1d(
            output_channels_sorted, variable_sorted, fill_value="extrapolate"
            )
    variable_interpolated = interp_function(fastJX_channels)

    return variable_interpolated


def plot_interpolation_comparison(output_channels, variable, methods=['linear', 'power_law']):
    """
    Plot comparison of different interpolation/extrapolation methods.
    """
    fastJX_channels = np.array([
        187.0, 191.0, 193.0, 196.0, 202.0, 208.0, 211.0, 214.0,
        261.0, 267.0, 277.0, 295.0, 303.0, 310.0, 316.0, 333.0,
        383.0, 599.0, 973.0, 1267.0, 1448.0, 1767.0, 2039.0, 2309.0,
        2748.0, 3404.0, 5362.0
    ])

    plt.figure(figsize=(12, 8))

    # Plot original data
    sorted_indices = np.argsort(output_channels)
    orig_wl = np.array(output_channels)[sorted_indices]
    orig_var = np.array(variable)[sorted_indices]

    plt.loglog(orig_wl, orig_var, 'ko-', label='Original measurements', markersize=8, linewidth=2)

    # Plot different methods
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, method in enumerate(methods):
        if method == 'original':
            # Original linear extrapolation
            interp_func = interpolate.interp1d(orig_wl, orig_var, fill_value="extrapolate")
            var_interp = interp_func(fastJX_channels)
            var_interp = np.maximum(var_interp, 0)  # Just clip negatives
            label = 'Original (linear extrap + clipping)'
        else:
            var_interp = interpolate_to_fastJX_improved(
                output_channels, variable, extrapolation_method=method
            )
            label = f'Improved ({method})'

        plt.loglog(fastJX_channels, var_interp, 'x--', color=colors[i],
                   label=label, markersize=6, linewidth=1.5)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('AOD')
    plt.title('Comparison of AOD Interpolation/Extrapolation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(1e-5, 1)
    plt.show()


def interp_test(output_channels, variable, interpolated_variable, title):
    # Define the specific FastJX channels (wavelengths)
    fastJX_channels = np.array(
                    [187.0, 191.0, 193.0, 196.0, 202.0, 208.0, 211.0, 214.0,
                     261.0, 267.0, 277.0, 295.0, 303.0, 310.0, 316.0, 333.0,
                     383.0, 599.0, 973.0, 1267.0, 1448.0, 1767.0, 2039.0, 2309.0,
                     2748.0, 3404.0, 5362.0]
                    )
    # sort output_channels and initial variable
    sorted_indices = np.argsort(output_channels)
    channels_sorted = np.array(output_channels)[sorted_indices]
    variable_sorted = np.array(variable)[sorted_indices]

    # Plot original data (output_channels vs variable)
    plt.figure(figsize=(10, 6))
    plt.plot(channels_sorted,
             variable_sorted,
             'o-', label='Original Data',
             color='blue')

    # Plot interpolated data (fastJX_channels vs interpolated_variable)
    plt.plot(fastJX_channels,
             interpolated_variable,
             'x--', label='Interpolated Data',
             color='red')

    # Add labels and title
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Variable')
    plt.title(f'Interpolation/Extrapolation Test ({title})')

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def sanitize_aerosol_name(aerosol_name):
    # Replace '/' with '_' and remove and '.'
    return aerosol_name.replace('/', '_').replace('.', '').replace(' ', '_')


def write_aerosol_properties(aerosol_name, AOD_values, SSA_values, G_values):
    # Output directory
    out_dir = './aerosol_fastJcsvFiles'

    # Create directory if does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # sanitize aerosol_name to create safe filenames
    safe_aerosol_name = sanitize_aerosol_name(aerosol_name)

    # Create filenames based on aerosol_name
    aod_filename = os.path.join(out_dir, f'AOD_{safe_aerosol_name}.csv')
    ssa_filename = os.path.join(out_dir, f'SSA_{safe_aerosol_name}.csv')
    g_filename = os.path.join(out_dir, f'G_{safe_aerosol_name}.csv')

    # Function to append values without overwriting the header
    def append_values(filename, values):
        file_exists = os.path.isfile(filename)

        # Open the file in append mode
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:  # If the file does not exist, write the header first
                writer.writerow(fastJX_channels)
            # Write the new set of values
            writer.writerow(values)

    # Append AOD values
    append_values(aod_filename, AOD_values)

    # Append SSA values
    append_values(ssa_filename, SSA_values)

    # Append G values
    append_values(g_filename, G_values)


def write_aerosol_heights(aerosol_name, rsp_height, hsrl_height):
    # Output directory
    out_dir = './aerosol_heights'

    # Create directory if does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Sanitize aerosol_name to create safe filenames
    safe_aerosol_name = sanitize_aerosol_name(aerosol_name)

    # Create filenames based on aerosol_name
    rsp_filename = os.path.join(out_dir, f'RSP_{safe_aerosol_name}_hts.csv')
    hsrl_filename = os.path.join(out_dir, f'HSRL_{safe_aerosol_name}_hts.csv')

    # Function to append values without overwriting the header
    def append_values(filename, values, header):
        file_exists = os.path.isfile(filename)

        # Open the file in append mode
        with open(filename, mode='a', newline='') as file:
            if not file_exists:
                file.write(header + "\n")

            # Check if 'values' is iteratable; if not, make it a list
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = [values]

            # Write the new set of values, one value per line
            for value in values:
                file.write(f"{value}\n")

    # Append RSP height values
    append_values(rsp_filename, rsp_height, f'RSP Aerosol Heights for {aerosol_name}')

    # Append HSRL height values
    append_values(hsrl_filename, hsrl_height, f'HSRL Aerosol Heights for {aerosol_name}')


def main():
    # Data directory
    data_directory = '/Users/adambell/Research/photochemistry/manuscript/collocationTest/Data/2020'

    # List of file paths for RSP and HSRL files (ensure they are in matching order)
    RSPfiles = [
        'ACTIVATE-RSP-AER_UC12_20200214_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200215_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200217_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200227_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200229_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200302_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200309_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200311_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200813_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200817_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200820_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200821_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200825_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200826_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200828_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200902_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200903_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200911_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200921_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200922_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200923_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200929_R03.h5',
        'ACTIVATE-RSP-AER_UC12_20200930_R03.h5',
    ]
    HSRLfiles = [
        'ACTIVATE-HSRL2_UC12_20200214_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200215_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200217_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200227_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200229_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200302_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200309_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200311_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200813_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200817_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200820_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200821_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200825_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200826_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200828_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200902_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200903_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200911_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200921_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200922_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200923_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200929_R4.h5',
        'ACTIVATE-HSRL2_UC12_20200930_R4.h5',
    ]

    # Prepend directory path to each file
    RSPfiles = [os.path.join(data_directory, f) for f in RSPfiles]
    HSRLfiles = [os.path.join(data_directory, f) for f in HSRLfiles]

    if len(RSPfiles) != len(HSRLfiles):
        print("The number of RSP and HSRL files must be the same.")
        return

    # Loop through each pair of RSP and HSRL files
    for rsp_file, hsrl_file in zip(RSPfiles, HSRLfiles):
        print(f"Processing RSP file: {rsp_file} and HSRL file: {hsrl_file}")

        # Step 1: Read variable "time" in UTC from RSP and convert to seconds since midnight
        rsp_time_utc = read_hdf5_variable(rsp_file, 'time')
        rsp_time_seconds = convert_time_to_seconds_utc(rsp_time_utc)

        # Step 2: Read variable "time" in seconds since midnight from HSRLfile
        hsrl_time_seconds = read_hdf5_variable(hsrl_file, 'time')

        # Step 3: Read latitude and longitude from both files
        rsp_latitude = read_hdf5_variable(rsp_file, 'lat')
        rsp_longitude = read_hdf5_variable(rsp_file, 'lon')
        hsrl_latitude = read_hdf5_variable(hsrl_file, 'lat')
        hsrl_longitude = read_hdf5_variable(hsrl_file, 'lon')

        # Step 4: Read RSP output channels, AOD, ssa, g
        output_channels = read_hdf5_variable(rsp_file, 'output_channels')
        aerosol_tau_f = read_hdf5_variable(rsp_file, 'aerosol_tau_f')
        aerosol_tau_c = read_hdf5_variable(rsp_file, 'aerosol_tau_c')
        aerosol_ssa_f = read_hdf5_variable(rsp_file, 'aerosol_ssa_f')
        aerosol_ssa_c = read_hdf5_variable(rsp_file, 'aerosol_ssa_c')
        aerosol_asymmetry_f = read_hdf5_variable(rsp_file, 'aerosol_asymmetry_f')
        aerosol_asymmetry_c = read_hdf5_variable(rsp_file, 'aerosol_asymmetry_c')
        aerosol_top_ht_rsp = read_hdf5_variable(rsp_file, 'aerosol_top_height')

        # Step 5: Read the Aerosol_ID and Altitude variables from HSRL
        aerosol_id = read_hdf5_variable(hsrl_file, 'DataProducts/Aerosol_ID')
        altitude = read_hdf5_variable(hsrl_file, 'DataProducts/Altitude').flatten()

        # Step 6: setup Aerosol ID to Type mapping
        aerosol_id_to_type = {
            1: 'Ice',
            2: 'Dusty Mix',
            3: 'Marine',
            4: 'Urban/Pollution',
            5: 'Smoke',
            6: 'Fresh Smoke',
            7: 'Pol. Marine',
            8: 'Dust',
            9: 'Untyped ambiguous',
            10: 'Untyped ambiguous'
        }

        # Step 7: Process all times in RSP file (loop through the times in RSP file)
        for idx, (rsp_time, rsp_lat, rsp_lon) in enumerate(zip(
                rsp_time_seconds, rsp_latitude, rsp_longitude
                )):
            # Find the closest time in HSRLfile
            closest_index = find_closest_time_index(rsp_time, hsrl_time_seconds)
            hsrl_time = hsrl_time_seconds[closest_index]

            # Calculate time difference
            delta_time = abs(rsp_time - hsrl_time)

            # Get corresponding latitude and longitude from HSRLfile
            hsrl_lat = hsrl_latitude[closest_index]
            hsrl_lon = hsrl_longitude[closest_index]

            # Calculate latitude and longitude differences
            delta_lat, delta_lon = abs(rsp_lat - hsrl_lat), abs(rsp_lon - hsrl_lon)

            # Skip to the next iteration if the minimum criteria are not met
            if delta_lat > min_delta_lat or delta_lon > min_delta_lon or \
                    delta_time > min_delta_time:
                continue  # Skip rest of loop and go to next iteration

            # Extract non-NaN Aerosol_ID vals and indices for the closest index
            aerosol_id_non_nan, non_nan_indices = \
                get_non_nan_values(aerosol_id[closest_index, :])

            # Calculate percentages of unique values
            unique_percentages = count_unique_values_percentage(aerosol_id_non_nan)

            # Check for percentages above the threshold
            if return_max:
                # Find the maximum percentage aerosol type that meets the threshold
                max_aerosol_type = None
                max_percentage = 0
                for aerosol_type, percentage in unique_percentages.items():
                    if percentage > threshold and percentage > max_percentage:
                        max_aerosol_type = aerosol_type
                        max_percentage = percentage

                if max_aerosol_type is not None:
                    aerosol_name = aerosol_id_to_type.get(
                            int(max_aerosol_type), "Unknown"
                            )
                    # print(f"Values exceeding threshold ({threshold}):")
                    print(f"Closest index: {closest_index},"
                          f"Aerosol ID: {max_aerosol_type},"
                          f"Percentage: {max_percentage:.2f},"
                          f"Type: {aerosol_name}")
                    aero_top_height = aerosol_top_ht_rsp[idx]
                    print(f"aerosol_top_height_RSP = {aero_top_height}")
                    print(f"aerosol_top_height_HSRL = {altitude[max(non_nan_indices)]}")

                    # Collect variables
                    aero_tau_f = aerosol_tau_f[idx, :]
                    aero_tau_c = aerosol_tau_c[idx, :]
                    aero_ssa_f = aerosol_ssa_f[idx, :]
                    aero_ssa_c = aerosol_ssa_c[idx, :]
                    aero_asymmetry_f = aerosol_asymmetry_f[idx, :]
                    aero_asymmetry_c = aerosol_asymmetry_c[idx, :]

                    # Interpolate values to FastJX wavelengths
                    # method = 'cubic'
                    # New method that works best is 'exponential'
                    # aerosol_tau_f_fastJX = interpolate_to_fastJX(output_channels, aero_tau_f, method)

                    # testing method 3
                    result = interpolate_to_fastJX_simple(output_channels, aero_tau_f)
                    print("\nFinal interpolated AOD values:")
                    fastJX_channels = np.array([
                        187.0, 191.0, 193.0, 196.0, 202.0, 208.0, 211.0, 214.0,
                        261.0, 267.0, 277.0, 295.0, 303.0, 310.0, 316.0, 333.0,
                        383.0, 599.0, 973.0, 1267.0, 1448.0, 1767.0, 2039.0, 2309.0,
                        2748.0, 3404.0, 5362.0
                    ])
                    
                    for i, (wl, aod) in enumerate(zip(fastJX_channels, result)):
                        print(f"{wl:6.1f} nm: {aod:.6f}")
                    
                    print(f"\nSummary:")
                    print(f"Minimum AOD: {np.min(result):.2e}")
                    print(f"Maximum AOD: {np.max(result):.2e}")
                    print(f"Negative values: {np.sum(result < 0)}")
                    # sys.exit()

                    # method = 'exponential'
                    # aerosol_tau_f_fastJX = interpolate_to_fastJX_improved(output_channels, aero_tau_f, extrapolation_method=method)
                    # print(f"output_channels = {output_channels}")
                    # print(f"aero_tau_f = {aero_tau_f}")
                    # print(f"aerosol_tau_interp = {aerosol_tau_f_fastJX}")
                    # sys.exit()

                    # methods_to_test = ['power_law', 'exponential', 'constant']
                    # for method in methods_to_test:
                    #     # aerosol_tau_f_fastJX = interpolate_to_fastJX_improved(
                    #     #         output_channels, aero_tau_f, extrapolation_method=method)
                    #     aerosol_tau_f_fastJX = interpolate_to_fastJX_improved(
                    #             output_channels, aero_ssa_f, extrapolation_method=method)

                    #     print(f"\n{method.upper()} method:")
                    #     print("FastJX wavelengths with potential negatives (>2264 nm):")
                    #     long_wl_indices = np.array([22, 23, 24, 25, 26])  # 2309, 2748, 3404, 5362 nm
                    #     long_wavelengths = [2309.0, 2748.0, 3404.0, 5362.0]

                    #     for i, wl in enumerate(long_wavelengths):
                    #         idx = 23 + i  # Starting from index 23 for 2309 nm
                    #         print(f"  {wl} nm: {aerosol_tau_f_fastJX[idx]:.6f}")

                    #     # Check for negative values
                    #     neg_count = np.sum(aerosol_tau_f_fastJX < 0)
                    #     print(f"  Negative values: {neg_count}")

                    # # Uncomment to see the plot comparison
                    # # plot_interpolation_comparison(output_channels, aero_tau_f,
                    # #                               ['original', 'power_law', 'exponential', 'constant'])
                    # plot_interpolation_comparison(output_channels, aero_ssa_f,
                    #                               ['original', 'power_law', 'exponential', 'constant'])
                    # sys.exit()

                    # aerosol_tau_f_fastJX = interpolate_to_fastJX_improved(output_channels, aero_tau_f, extrapolation_method=method)
                    # aerosol_tau_c_fastJX = interpolate_to_fastJX_improved(output_channels, aero_tau_c, extrapolation_method=method)
                    # aerosol_ssa_f_fastJX = interpolate_to_fastJX_improved(output_channels, aero_ssa_f, extrapolation_method=method)
                    # aerosol_ssa_c_fastJX = interpolate_to_fastJX_improved(output_channels, aero_ssa_c, extrapolation_method=method)
                    # aerosol_asymmetry_f_fastJX = interpolate_to_fastJX_improved(output_channels, aero_asymmetry_f, extrapolation_method=method)
                    # aerosol_asymmetry_c_fastJX = interpolate_to_fastJX_improved(output_channels, aero_asymmetry_c, extrapolation_method=method)

                    aerosol_tau_f_fastJX = interpolate_to_fastJX_simple(output_channels, aero_tau_f)
                    aerosol_tau_c_fastJX = interpolate_to_fastJX_simple(output_channels, aero_tau_c)
                    aerosol_ssa_f_fastJX = interpolate_to_fastJX_simple(output_channels, aero_ssa_f)
                    aerosol_ssa_c_fastJX = interpolate_to_fastJX_simple(output_channels, aero_ssa_c)
                    aerosol_asymmetry_f_fastJX = interpolate_to_fastJX_simple(output_channels, aero_asymmetry_f)
                    aerosol_asymmetry_c_fastJX = interpolate_to_fastJX_simple(output_channels, aero_asymmetry_c)

                    # Write values to csvfile for FastJX reading
                    write_aerosol_properties(aerosol_name, aerosol_tau_f_fastJX,
                                             aerosol_ssa_f_fastJX,
                                             aerosol_asymmetry_f_fastJX)
                    write_aerosol_heights(aerosol_name, aero_top_height,
                                          altitude[max(non_nan_indices)])

            else:
                # Print all aerosol types that meet the threshold, and keep track
                # of printed aerosol types to avoid duplicates
                printed_types = set()
                print(f"Values exceeding threshold ({threshold}):")
                for aerosol_type, percentage in unique_percentages.items():
                    if percentage > threshold and aerosol_type not in printed_types:
                        aerosol_name = aerosol_id_to_type.get(
                                int(aerosol_type), "Unknown")
                        print(f"Closest index: {closest_index},"
                              f"Aerosol ID: {aerosol_type},"
                              f"Percentage: {percentage:.2f},"
                              f"Type: {aerosol_name}\n")
                        # Mark this aerosol type as printed
                        printed_types.add(aerosol_type)


if __name__ == "__main__":
    main()
