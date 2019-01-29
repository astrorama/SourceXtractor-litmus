#!/usr/bin/env python3
#
# Compare a stuff simulation and a sextractor output, and generate
# stats that can be used as baseline for the tests
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from astropy.table import Table

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from util import stuff


def print_stats(label, values):
    """
    Print basic statistics about the values
    """
    print(label)
    print(f'\tMin:    {np.min(values)}')
    print(f'\tMax:    {np.max(values)}')
    print(f'\tMean:   {np.mean(values)}')
    print(f'\tStdDev: {np.std(values)}')
    print(f'\tsqrt(sum(squared)): {np.sqrt(np.sum(values**2))}')

# Options
parser = ArgumentParser()
parser.add_argument('catalog', type=str, help='sextractor catalog')
parser.add_argument('stuff', type=str, help='Stuff original simulation')
parser.add_argument('--plot', action='store_true', default=False, help='Show histogram')
parser.add_argument('--world-centroid-alpha', default='ALPHA_SKY', help='Right ascension')
parser.add_argument('--world-centroid-delta', default='DELTA_SKY', help='Declination')
parser.add_argument('--magnitude-zeropoint', type=float, default=26., help='Magnitude zeropoint')
parser.add_argument('--exposure', type=float, default=300., help='Exposure time')
parser.add_argument(
    '--flux-column', action='append', nargs='+',
    help='Columns that contain fluxes'
)

args = parser.parse_args()

catalog = Table.read(args.catalog)
stars, galaxies = stuff.parse_stuff_list(args.stuff)

all_mags = np.append(stars.mag, galaxies.mag)

print(f'{len(catalog)} sources identified by sextractor')
print(f'{len(stars) + len(galaxies)} on the original stuff simulation')

print("Building KDTree")
kdtree, n_stars, n_galaxies = stuff.index_sources(stars, galaxies)

print("Doing cross-matching")
closest = stuff.get_closest(catalog, kdtree, alpha=args.world_centroid_alpha, delta=args.world_centroid_delta)

print_stats('Distance', np.abs(closest['dist']))
mag_diffs = {}

for col in args.flux_column:
    mag = stuff.flux2mag(catalog[col], args.magnitude_zeropoint, args.exposure)
    mag_diffs[col] = mag[closest['catalog']] - all_mags[closest['source']]
    print_stats(col, mag_diffs[col])

# Show histogram
if args.plot:
    plt.subplot(1, 2, 1)
    plt.title('Distances')
    plt.hist(closest['dist'], bins=50)
    plt.subplot(1, 2, 2)
    plt.title('Fluxes')
    for col in args.flux_column:
        plt.hist(mag_diffs[col], bins=50, alpha=0.5, label=col)
    plt.legend()
    plt.show()
