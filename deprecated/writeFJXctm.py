#!/usr/bin/env python3
import sys
import numpy as np


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


def write_CTM_GrdCld(aerosol_height):
    """
    This function currently assumes 1 singular aerosol layer top height,
    and we don't take into account layer thickness. In the future we likely
    should account for specific layer thickness.
    """

    filename_atmo = 'CTM_atmo.dat'

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

    print(f'aer_p = {aer_p}, aer_p shape = {aer_p.shape}')
    print(f'aer_nda = {aer_nda}, aer_nda shape = {aer_nda.shape}')
    print(f'alts = {alts}')

    # Start with only 1 mode (i.e., aerosol type)
    z1 = find_nearest(alts, aerosol_height[0] * 100000)
    idx1 = alts <= z1
    z2 = 0
    idx2 = alts < z2

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


# Initialize variables
aer_z = np.zeros(2)
aer_z[0] = 1.5

# Call function to write CTM_GrdCld
write_CTM_GrdCld(aer_z)
