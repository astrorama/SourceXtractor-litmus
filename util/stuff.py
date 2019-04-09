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
    NONE = 0  # No flag is set
    BIASED = 1 << 0  # The object has bad pixels
    BLENDED = 1 << 1  # The object was originally blended with another one.
    SATURATED = 1 << 2  # At least one pixel of the object is saturated.
    BOUNDARY = 1 << 3  # The object is truncates (to close to an image boundary).
    NEIGHBORS = 1 << 4  # The object has neighbors, bright and close enough
    OUTSIDE = 1 << 5  # The object is completely outside of the measurement frame


class Sex2SourceFlags(IntFlag):
    """
    SExtractor 2 flags
    """
    NONE = 0  # No flag is set
    # 1) The object has neighbors, bright and close enough to
    #    significantly bias the photometry, or bad pixels
    #    (more than 10% of the integrated area affected).
    BIASED = 1 << 0
    # 2) The object was originally blended with another one.
    BLENDED = 1 << 1
    # 4) At least one pixel of the object is saturated (or very close to).
    SATURATED = 1 << 2
    # 8) The object is truncates (to close to an image boundary).
    BOUNDARY = 1 << 3
    # 16) Object's aperture data are incomplete or corrupted.
    APERTURE_INCOMPLETE = 1 << 4
    # 32) Object's isophotal data are incomplete or corrupted.
    ISOPHOTAL_INCOMPLETE = 1 << 5
    # 64) A memory overflow occurred during deblending.
    DEBLENDING_OVERFLOW = 1 << 6
    # 128) A memory overflow occurred during extraction.
    EXTRACTION_OVERFLOW = 1 << 7


class Simulation(object):
    def __init__(self, path, mag_zeropoint, exposure):
        """
        Parses a Stuff input file. See https://www.astromatic.net/software/stuff
        :param path:
            Path to the .list simulation data
        :param mag_zeropoint:
            Used to compute the flux
        :param exposure:
            Used to compute the flux
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

        self.__stars = np.recarray((len(stars_raw),), dtype=[
            ('ra', float), ('dec', float), ('mag', float), ('flux', float)
        ])
        self.__galaxies = np.recarray((len(galaxies_raw),), dtype=[
            ('ra', float), ('dec', float), ('mag', float), ('flux', float),
            ('bt_ratio', float), ('bulge', float), ('bulge_aspect', float), ('disk', float), ('disk_aspect', float),
            ('redshift', float), ('type', float)
        ])

        for i, s in enumerate(stars_raw):
            self.__stars[i].ra, self.__stars[i].dec, self.__stars[i].mag = float(s[1]), float(s[2]), float(s[3])
        self.__stars.flux = exposure * np.power(10, (self.__stars.mag - mag_zeropoint) / -2.5)

        for i, g in enumerate(galaxies_raw):
            self.__galaxies[i].ra, self.__galaxies[i].dec = float(g[1]), float(g[2])
            self.__galaxies[i].mag = float(g[3])
            self.__galaxies[i].bt_ratio = float(g[4])
            self.__galaxies[i].bulge, self.__galaxies[i].bulge_aspect = float(g[5]), float(g[6])
            self.__galaxies[i].disk, self.__galaxies[i].disk_aspect = float(g[8]), float(g[9])
            self.__galaxies[i].redshift = float(g[11])
            self.__galaxies[i].type = float(g[12])
        self.__galaxies.flux = exposure * np.power(10, (self.__galaxies.mag - mag_zeropoint) / -2.5)

        all_coords = np.column_stack([
            np.append(self.__stars.ra, self.__galaxies.ra),
            np.append(self.__stars.dec, self.__galaxies.dec)
        ])
        self.__kdtree = KDTree(all_coords)
        self.__all_mags = np.append(self.__stars.mag, self.__galaxies.mag)

    @property
    def stars(self):
        return self.__stars

    @property
    def galaxies(self):
        return self.__galaxies

    @property
    def magnitudes(self):
        """
        :return: All the magnitudes on the simulation: first stars, then galaxies.
        """
        return self.__all_mags

    def get_star_count(self):
        """
        :return: Number of stars on the simulation.
        """
        return len(self.__stars)

    def get_galaxy_count(self):
        """
        :return: Number of galaxies on the simulation.
        """
        return len(self.__galaxies)

    def get_closest(self, alpha, delta):
        """
        Find the closest source to the catalog entries. This can be used to cross-relate
        the entries from the Stuff simulation and a catalog
        :param alpha:
            Alpha coordinates
        :param delta:
            Delta coordinates
        :return:
            A numpy array with the columns:
                * distance: distance to closest source
                * catalog: corresponding index on the catalog
                * source: corresponding index on the simulation list
                * magnitude: magnitude of the closest source
                * is_star: True if the closest source is a star
                * is_galaxy: True if the closest source is a galaxy
        """
        assert len(alpha) == len(delta)
        closest = np.recarray((len(alpha),), dtype=[
            ('distance', float), ('catalog', int), ('source', int), ('magnitude', float),
            ('is_star', bool), ('is_galaxy', bool)
        ])

        closest.catalog[:] = np.arange(0, len(alpha))
        closest.distance[:], closest.source[:] = self.__kdtree.query(np.column_stack([alpha, delta]))
        closest.magnitude[:] = self.__all_mags[closest.source]
        closest.is_star[:] = closest.source < self.get_star_count()
        closest.is_galaxy[:] = np.logical_not(closest.is_star[:])
        return closest


def flux2mag(fluxes, mag_zeropoint=26., exposure=300.):
    """
    Convert flux to magnitude
    """
    return mag_zeropoint - 2.5 * np.log10(fluxes / exposure)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('Required one (and only one) .list file as a parameter')
        sys.exit(-1)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    simulation = Simulation(sys.argv[1])

    import matplotlib.pyplot as plt

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.hist(simulation.galaxies.redshift)
    plt.title('Galaxy redshift')

    plt.subplot(2, 2, 2)
    plt.scatter(simulation.galaxies.ra, simulation.galaxies.dec, c=simulation.galaxies.mag)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.colorbar()
    plt.title('Galaxy magnitudes')

    plt.subplot(2, 2, 4)
    plt.scatter(simulation.stars.ra, simulation.stars.dec, c=simulation.stars.mag)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.colorbar()
    plt.title('Star magnitudes')

    plt.tight_layout()
    plt.show()
