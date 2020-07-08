#!/bin/env python3
import os
import numpy as np
from argparse import ArgumentParser
from astropy.table import Table, Column
from stuff import Simulation
import astropy.units as u

parser = ArgumentParser()
parser.add_argument('-m', '--mag-zeropoint', type=float, default=32.19)
parser.add_argument('-e', '--exposure', type=float, default=300.0)
parser.add_argument('-o', '--output', metavar='OUTPUT', type=str, default=None)
parser.add_argument('input', metavar='FILE', type=str)

args = parser.parse_args()

if args.output is None:
    path = os.path.dirname(args.input)
    filename = os.path.basename(args.input)
    filename = os.path.splitext(filename)[0]
    args.output = os.path.join(path, filename + '.sim.fits')

print(args.input, '>>', args.output)

sim = Simulation(args.input, args.mag_zeropoint, args.exposure)

columns = []
# Shared attributes
for k, u in [('ra', u.deg), ('dec', u.deg), ('mag', u.mag), ('flux', u.count)]:
    columns.append(
        Column(np.concatenate([sim.galaxies[k], sim.stars[k]]), name=k, unit=u)
    )

# Galaxies only
for k in ['bt_ratio', 'bulge', 'bulge_aspect', 'disk', 'disk_aspect', 'redshift']:
    columns.append(
        Column(np.concatenate([sim.galaxies[k], np.full(len(sim.stars), np.nan)]), name=k)
    )

# Flag
columns.append(
    Column(np.concatenate([
        np.ones(len(sim.galaxies), dtype=np.int),
        np.zeros(len(sim.stars), dtype=np.int)
    ]), name='galaxy')
)

# Write
t = Table(columns)
t.write(args.output, overwrite=True)

