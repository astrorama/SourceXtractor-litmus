#!/usr/bin/env python3
#
# Stuff .list parser

import numpy as np
import logging
from enum import IntFlag

from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

class SourceFlags(IntFlag):
    """
    SExtractor++ flags
    """
    NONE      = 0 # No flag is set
    BIASED    = 1 << 0 # The object has bad pixels
    BLENDED   = 1 << 1 # The object was originally blended with another one.
    SATURATED = 1 << 2 # At least one pixel of the object is saturated.
    BOUNDARY  = 1 << 3 # The object is truncates (to close to an image boundary).
    NEIGHBORS = 1 << 4 # The object has neighbors, bright and close enough
    OUTSIDE   = 1 << 5 # The object is completely outside of the measurement frame


def parse_stuff_list(path):
    """
    Parses a Stuff input file. See https://www.astromatic.net/software/stuff
    :return: A tuple with the list of stars, and the list of galaxies
    """
    logger.debug(f'Loading stuff list from {path}')
    stars_raw, galaxies_raw = [], []
    with open(path, 'r') as lst:
        for entry in map(str.split, map(str.strip, lst.readlines())):
            if entry[0] == '100':
                stars_raw.append(entry)
            elif entry[0] == '200':
                galaxies_raw.append(entry)
            else:
                raise Exception(f'Unexpected type code {entry[0]}')

    logger.debug(f'Loaded {len(stars_raw)} stars')
    logger.debug(f'Loaded {len(galaxies_raw)} galaxies')

    stars = np.recarray((len(stars_raw),), dtype=[
        ('ra', float), ('dec', float), ('mag', float)
    ])
    galaxies = np.recarray((len(galaxies_raw),), dtype=[
        ('ra', float), ('dec', float), ('mag', float), ('bt_ratio', float),
        ('bulge', float), ('bulge_aspect', float), ('disk', float), ('disk_aspect', float),
        ('redshift', float), ('type', float)
    ])

    for i, s in enumerate(stars_raw):
        stars[i].ra, stars[i].dec, stars[i].mag = float(s[1]), float(s[2]), float(s[3])

    for i, g in enumerate(galaxies_raw):
        galaxies[i].ra, galaxies[i].dec, galaxies[i].mag = float(g[1]), float(g[2]), float(g[3])
        galaxies[i].bt_ratio = float(g[4])
        galaxies[i].bulge, galaxies[i].bulge_aspect = float(g[5]), float(g[6])
        galaxies[i].disk, galaxies[i].disk_aspect = float(g[8]), float(g[9])
        galaxies[i].redshift = float(g[11])
        galaxies[i].type = float(g[12])

    return stars, galaxies


def index_sources(stars, galaxies):
    """
    Creates a KDTree with the stars and galaxies coordinates
    :param stars:
        Star list (must have .ra and .dec)
    :param galaxies:
        Galaxy list (must have .ra and .dec)
    :return:
        A tuple (kdtree, number of stars, number of galaxies)
        When querying the kdtree, it will return the position on the original list.
        Stars go first, so if the position < number of stars, then it is a star,
        otherwise, it is a galaxy (with position index - number of stars)
    """
    stars_coords = np.stack([stars.ra, stars.dec]).T
    galaxies_coords = np.stack([galaxies.ra, galaxies.dec]).T
    all_coords = np.append(stars_coords, galaxies_coords, axis=0)
    all_kdtree = KDTree(all_coords)
    n_stars = len(stars)
    n_galaxies = len(galaxies)
    return all_kdtree, n_stars, n_galaxies


def get_closest(catalog, kdtree, alpha='world_centroid_alpha', delta='world_centroid_delta'):
    """
    Find the closest source to the catalog entries. This can be used to cross-relate
    the entries from the Stuff simulation and a catalog
    :param catalog:
        A table that can be accessed column-wise
    :param alpha:
        Table column for the alpha coordinate
    :param delta:
        Table column for the delta coordinate
    :return:
        A dictionary where each value is a list of the same size, and each position correspond to the same entry:
            * dist: distance to closest source
            * catalog: corresponding index on the catalog
            * source: corresponding index on the stuff list
    """
    distances = []
    index_c = []
    index_s = []
    for i, e in enumerate(catalog):
        d, s = kdtree.query([e['world_centroid_alpha'], e['world_centroid_delta']], 1)
        distances.append(d)
        index_c.append(i)
        index_s.append(s)
    return {
        'dist': np.array(distances),
        'catalog': np.array(index_c),
        'source': np.array(index_s),
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('Required one (and only one) .list file as a parameter')
        sys.exit(-1)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    stars, galaxies = parse_stuff_list(sys.argv[1])

    import matplotlib.pyplot as plt

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.hist(galaxies.redshift)
    plt.title('Galaxy redshift')

    plt.subplot(2, 2, 2)
    plt.scatter(galaxies.ra, galaxies.dec, c=galaxies.mag)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.colorbar()
    plt.title('Galaxy magnitudes')

    plt.subplot(2, 2, 4)
    plt.scatter(stars.ra, stars.dec, c=stars.mag)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.colorbar()
    plt.title('Star magnitudes')

    plt.tight_layout()
    plt.show()
