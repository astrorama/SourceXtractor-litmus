#!/usr/bin/env python3
import itertools
import logging
import os
import platform
import subprocess
import sys
from argparse import ArgumentParser

from astropy.table import Table

from util import stuff, plot


def open_with_system(path):
    """
    Open the file with the application configured on the system
    """
    if platform.system() == 'Darwin':
        subprocess.call(['open', path])
    elif platform.system() == 'Linux':
        subprocess.call(['xdg-open', path])


if __name__ == '__main__':
    log = logging.getLogger()
    log_handler = logging.StreamHandler(sys.stderr)
    log.addHandler(log_handler)

    log.setLevel(logging.INFO)
    log_handler.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--output', '-o', type=str, metavar='FILE', default='cross-validation.pdf',
                        help='Report filename')
    parser.add_argument('--detection-image', '-i', metavar='FILE', required=True, help='Detection image')
    parser.add_argument('--weight-image', '-w', metavar='FILE', required=True, help='Weight image')
    parser.add_argument('--stuff', '-s', type=str, metavar='FILE', required=True, help='Stuff catalog')
    parser.add_argument('--mag-zeropoint', type=float, default=26., help='Magnitude zeropoint for the Stuff catalog')
    parser.add_argument('--exposure', type=float, default=300, help='Exposure time for the Stuff catalog')
    parser.add_argument('--open', action='store_true', help='Open the report after creation')
    parser.add_argument('--pixel-x', type=str, default='mf_x', help='Column with the X coordinate')
    parser.add_argument('--pixel-y', type=str, default='mf_y', help='Column with the Y coordinate')
    parser.add_argument('--wc-alpha', type=str, default='world_centroid_alpha',
                        help='World coordinates Right Ascension')
    parser.add_argument('--wc-delta', type=str, default='world_centroid_delta', help='Wold coordinate Declination')
    parser.add_argument('--magnitude', type=str, default='mf_mag_r', help='Magnitude')
    parser.add_argument('--magnitude-error', type=str, default='mf_mag_r_err', help='Magnitude error')
    parser.add_argument('catalog', type=str, nargs='+', metavar='CATALOG', help='Catalogs to cross-match')

    args = parser.parse_args()

    log.info(f'Loading Stuff reference {args.stuff}')
    reference = stuff.Simulation(args.stuff, args.mag_zeropoint, args.exposure)

    catalogs = []
    for catalog_path in args.catalog:
        logging.info(f'Loading catalog {catalog_path}')
        catalogs.append((os.path.basename(catalog_path), Table.read(catalog_path)))

    logging.info(f'Loading detection image {args.detection_image} with weight {args.weight_image}')
    image = plot.Image(args.detection_image, args.weight_image)

    log.info(f'Creating report {args.output}')
    with plot.Report(args.output) as report:
        # Location
        markers = itertools.cycle('1234')
        loc_map = plot.Location(image, reference)
        for catalog_name, catalog in catalogs:
            loc_map.add(catalog_name, catalog, args.pixel_x, args.pixel_y, marker=next(markers))
        report.add(loc_map)

        # Distance plot
        markers = itertools.cycle('1234')
        distance = plot.Distances(image, reference)
        for catalog_name, catalog in catalogs:
            distance.add(catalog_name, catalog, args.pixel_x, args.pixel_y, marker=next(markers))
        report.add(distance)

        # Completeness
        completeness = plot.Completeness(image, reference)
        for catalog_name, catalog in catalogs:
            completeness.add(catalog_name, catalog, args.pixel_x, args.pixel_y, args.magnitude)
        report.add(completeness)

        # Histogram
        histogram = plot.Histogram(image, reference)
        for catalog_name, catalog in catalogs:
            histogram.add(catalog_name, catalog, args.wc_alpha, args.wc_delta, args.magnitude)
        report.add(histogram)

        # Magnitude plot
        markers = itertools.cycle('1234')
        mag = plot.Magnitude(os.path.basename(args.stuff), reference)
        for catalog_name, catalog in catalogs:
            mag.add(
                catalog_name, catalog,
                args.wc_alpha, args.wc_delta,
                args.magnitude, args.magnitude_error,
                marker=next(markers)
            )
        report.add(mag)

    if args.open:
        open_with_system(args.output)
