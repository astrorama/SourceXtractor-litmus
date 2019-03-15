from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from util.image import Image


class CrossValidation(object):
    class CrossValidationResult(object):
        def __init__(self, stars_found, stars_not_found, stars_recall, stars_catalog,
                     galaxies_found, galaxies_not_found, galaxies_recall, galaxies_catalog,
                     misids):
            self.stars_found = stars_found
            self.stars_not_found = stars_not_found
            self.stars_recall = stars_recall
            self.stars_catalog = stars_catalog
            self.galaxies_found = galaxies_found
            self.galaxies_not_found = galaxies_not_found
            self.galaxies_recall = galaxies_recall
            self.galaxies_catalog = galaxies_catalog
            self.misids = misids

        @property
        def all_catalog(self):
            return np.append(self.stars_catalog, self.galaxies_catalog)

        @property
        def all_magnitudes(self):
            return np.append(self.stars_found.mag, self.galaxies_found.mag)

    def __init__(self, image, simulation, max_dist=0.5, bin_size=1):
        if isinstance(image, str) or isinstance(image, Path):
            self.__image = Image(image)
        else:
            self.__image = image

        self.__max_dist = max_dist

        self.stars = self.__image.get_contained_sources(
            simulation.stars.ra, simulation.stars.dec, mag=simulation.stars.mag
        )
        self.galaxies = self.__image.get_contained_sources(
            simulation.galaxies.ra, simulation.galaxies.dec, mag=simulation.galaxies.mag
        )
        self.__all_mag = np.append(self.stars.mag, self.galaxies.mag)

        all_x = np.append(self.stars.x, self.galaxies.x)
        all_y = np.append(self.stars.y, self.galaxies.y)

        self.__kdtree = KDTree(np.column_stack([all_x, all_y]))

        all_mags = np.append(self.stars.mag, self.galaxies.mag)
        self.__min_mag = np.floor(np.min(all_mags))
        self.__max_mag = np.ceil(np.max(all_mags))
        self.__bin_size = bin_size
        self.__edges = np.arange(self.__min_mag - bin_size / 2., self.__max_mag + bin_size / 2., bin_size)

        self.stars_bins, _ = np.histogram(self.stars.mag, bins=self.__edges)
        self.galaxies_bins, _ = np.histogram(self.galaxies.mag, bins=self.__edges)

    @property
    def bin_centers(self):
        return self.__bin_size / 2. * (self.__edges[1:] + self.__edges[:-1])

    def __call__(self, x, y, mag=None):
        nstars = len(self.stars)
        ngalaxies = len(self.galaxies)

        d, i = self.__kdtree.query(
            np.column_stack([x, y])
        )
        dist_filter = (d <= self.__max_dist)

        hits = np.recarray((len(x),), dtype=[('source', int), ('catalog', int), ('distance', float)])
        hits['source'] = i
        hits['catalog'] = np.arange(len(x))
        hits['distance'] = d
        hits = hits[dist_filter]

        real_found = np.unique(hits['source'])
        real_catalog = []
        for r in real_found:
            real_catalog.append(np.sort(hits[hits['source'] == r], order='distance')['catalog'][0])
        real_catalog = np.asarray(real_catalog)

        # Found contains now the index of the "real" stars and galaxies with at least one match
        # If the index is < len(self.__stars), it is a star
        stars_found = real_found[real_found < nstars]
        stars_catalog = real_catalog[real_found < nstars]
        stars_not_found = np.setdiff1d(np.arange(nstars), stars_found)
        stars_hist, _ = np.histogram(self.stars.mag[stars_found], bins=self.__edges)
        stars_recall = np.divide(
            stars_hist, self.stars_bins,
            out=np.zeros(stars_hist.shape), where=self.stars_bins != 0
        )

        # If the index is > len(self.__stars, it is a galaxy)
        galaxies_found = real_found[real_found >= nstars] - nstars
        galaxies_catalog = real_catalog[real_found >= nstars]
        galaxies_not_found = np.setdiff1d(np.arange(ngalaxies), galaxies_found)
        galaxies_hist, _ = np.histogram(self.galaxies.mag[galaxies_found], bins=self.__edges)
        galaxies_recall = np.divide(
            galaxies_hist, self.galaxies_bins,
            out=np.zeros(galaxies_hist.shape), where=self.galaxies_bins != 0
        )

        # Detections that are too far from any "real" source
        # We show them binned by measured, and by nearest
        if mag is not None:
            bad_filter = (d >= self.__max_dist)
            bad_mag = mag[bad_filter]
            bad_hist, _ = np.histogram(bad_mag, bins=self.__edges)
            sum_hist = (galaxies_hist + stars_hist + bad_hist)
            misids = np.divide(
                bad_hist, sum_hist,
                out=np.zeros(bad_hist.shape), where=sum_hist != 0
            )
        else:
            misids = None

        return CrossValidation.CrossValidationResult(
            self.stars[stars_found], self.stars[stars_not_found], stars_recall, stars_catalog,
            self.galaxies[galaxies_found], self.galaxies[galaxies_not_found], galaxies_recall, galaxies_catalog,
            misids
        )
