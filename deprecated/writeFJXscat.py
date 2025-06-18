#!/usr/bin/env python3
import sys
import os
import numpy as np
import csv
import pandas as pd


debug = 2


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
# MAIN PROGRAM
# =============================================================================
def main():

    # Define directories containing csv files
    csvDir_aerosolProps = '/Users/adambell/Research/photochemistry/manuscript/multi-run/aerosol_fastJcsvFiles'
    csvDir_aerosolHeight = '/Users/adambell/Research/photochemistry/manuscript/multi-run/aerosol_heights'

    # Define individual property file names (will be done dynamically in
    # future based on the aerosol type)
    csvName_AOD = 'AOD_Urban_Pollution.csv'
    csvName_SSA = 'SSA_Urban_Pollution.csv'
    csvName_G = 'G_Urban_Pollution.csv'
    csvName_height = 'HSRL_Urban_Pollution_hts.csv'

    # Get full file paths
    file_name_AOD = os.path.join(csvDir_aerosolProps, csvName_AOD)
    file_name_SSA = os.path.join(csvDir_aerosolProps, csvName_SSA)
    file_name_G = os.path.join(csvDir_aerosolProps, csvName_G)
    file_name_height = os.path.join(csvDir_aerosolHeight, csvName_height)

    # read aod, ssa, g, aerosol layer height
    wavelengths, aod = read_aerosol_property_data(file_name_AOD)
    wavelengths, ssa = read_aerosol_property_data(file_name_SSA)
    wavelengths, g = read_aerosol_property_data(file_name_G)
    aerosol_heights, header = read_aerosol_heights(file_name_height)

    if debug > 1:
        print(f"file_name_AOD = {file_name_AOD}")
        print(f"file_name_SSA = {file_name_SSA}")
        print(f"file_name_G = {file_name_G}")
        print(f"wavelengths = {wavelengths}")
        print(f"aod shape = {aod.shape}")
        print(f"aod[0, :] = {aod[0, :]}")
        print(f"ssa shape = {ssa.shape}")
        print(f"ssa[0, :] = {ssa[0, :]}")
        print(f"g shape = {g.shape}")
        print(f"g[0, :] = {g[0, :]}")
        print(f"aerosol_height shape = {aerosol_heights.shape}")
        print(f"aerosol_height[0] = {aerosol_heights[0]}")
        print(f"aerosol_height header = {header}")

    # Write the fast-J scat file (for all runs we will have to loop through all
    # aod, ssa, g and aerosol height values and make an input file for each.
    write_JX_scatFile(wavelengths, aod[0, :], aod[0, :], ssa[0, :], ssa[0, :], g[0, :], g[0, :])

    # Write CRM_GrdCld file (again will need to be looped in full run)
    write_CTM_GrdCld(aerosol_heights[0])

    # Try to run fast-J
    


if __name__ == '__main__':
    main()
