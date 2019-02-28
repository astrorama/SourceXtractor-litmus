import abc
import logging
from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import colors

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
from scipy.stats import ks_2samp

from util import stuff
from . import get_column

_page_size = (11.7, 8.3)
_img_cmap = plt.get_cmap('Greys_r')
_img_norm = colors.SymLogNorm(1000, linscale=10)

_flag_style = {
    stuff.SourceFlags.BIASED: ('red', '1'),
    stuff.SourceFlags.BLENDED: ('blue', '2'),
    stuff.SourceFlags.SATURATED: ('orange', '+'),
    stuff.SourceFlags.BOUNDARY: ('pink', '3'),
    stuff.SourceFlags.NEIGHBORS: ('cyan', '4'),
    stuff.SourceFlags.OUTSIDE: ('white', 'x')
}
_sex2_flag_style = {
    stuff.Sex2SourceFlags.BIASED: ('red', '1'),
    stuff.Sex2SourceFlags.BLENDED: ('blue', '2'),
    stuff.Sex2SourceFlags.SATURATED: ('orange', '+'),
    stuff.Sex2SourceFlags.BOUNDARY: ('pink', '3'),
    stuff.Sex2SourceFlags.APERTURE_INCOMPLETE: ('skyblue', '4'),
    stuff.Sex2SourceFlags.ISOPHOTAL_INCOMPLETE: ('darkcyan', '1'),
    stuff.Sex2SourceFlags.DEBLENDING_OVERFLOW: ('crimson', 'D'),
    stuff.Sex2SourceFlags.EXTRACTION_OVERFLOW: ('crimson', 'X'),
}


def _get_sources_within_image(img, wcs, ra, dec, mag):
    w, h = img.shape
    pix_coord = wcs.all_world2pix(ra, dec, 1)
    # Open a bit, since the center may be outside of the image, but have an effect
    inside_image = (pix_coord[0] >= -10) & (pix_coord[0] < w + 10) & (pix_coord[1] >= -10) & (pix_coord[1] < h + 10)
    pix_x = pix_coord[0][inside_image]
    pix_y = pix_coord[1][inside_image]
    mag = mag[inside_image]
    return pix_x, pix_y, mag


class Plot(object):
    @abc.abstractmethod
    def get_figures(self):
        pass


class Report(object):
    def __init__(self, path):
        self.__pdf = PdfPages(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.__pdf.close()

    def add(self, plot):
        assert isinstance(plot, Plot)
        for fig in plot.get_figures():
            self.__pdf.savefig(fig)
            plt.close(fig)


class Location(Plot):

    def __init__(self, image, simulation):
        super(Location, self).__init__()
        hdu = fits.open(image)[0]
        self.__image = hdu.data
        self.__wcs = WCS(hdu.header)
        self.__figure = plt.figure(figsize=_page_size)
        self.__ax = self.__figure.add_subplot(1, 1, 1)
        self.__ax.set_title('Location')
        self.__ax.imshow(self.__image, cmap=_img_cmap, norm=deepcopy(_img_norm))
        # Stars
        mag_cmap = plt.get_cmap('magma_r')
        stars_x, stars_y, stars_mag = _get_sources_within_image(
            self.__image, self.__wcs, simulation[0].ra, simulation[0].dec, simulation[0].mag
        )
        self.__ax.scatter(
            stars_x, stars_y, c=stars_mag, marker='o',
            cmap=mag_cmap
        )

        galaxies_x, galaxies_y, galaxies_mag = _get_sources_within_image(
            self.__image, self.__wcs, simulation[1].ra, simulation[1].dec, simulation[1].mag
        )
        cax = self.__ax.scatter(
            galaxies_x, galaxies_y, c=galaxies_mag, marker='h',
            cmap=mag_cmap
        )
        cbar = self.__figure.colorbar(cax)
        cbar.ax.set_ylabel('Magnitude (truth)')

    def add(self, label, catalog, x_col, y_col, marker=None):
        self.__ax.scatter(catalog[x_col], catalog[y_col], marker=marker, label=label, alpha=0.8, )

    def get_figures(self):
        self.__ax.legend()
        return [self.__figure]


class Distances(Plot):
    def __init__(self, image, simulation):
        super(Distances, self).__init__()
        hdu = fits.open(image)[0]
        image = hdu.data
        wcs = WCS(hdu.header)

        stars_x, stars_y, stars_mag = _get_sources_within_image(
            image, wcs, simulation[0].ra, simulation[0].dec, simulation[0].mag
        )
        galaxies_x, galaxies_y, galaxies_mag = _get_sources_within_image(
            image, wcs, simulation[1].ra, simulation[1].dec, simulation[1].mag
        )

        self.__x = np.append(stars_x, galaxies_x)
        self.__y = np.append(stars_y, galaxies_y)
        self.__mag = np.append(stars_mag, galaxies_mag)
        self.__kdtree = KDTree(np.column_stack([self.__x, self.__y]))
        self.__entries = []

    def add(self, label, catalog, x_col, y_col, marker):
        self.__entries.append((label, catalog[x_col], catalog[y_col], marker))

    def _plot_distances(self):
        fig = plt.figure(figsize=_page_size)
        nrows = len(self.__entries)
        bins = np.arange(-10.25, 10.25, 0.5)
        for i, (label, x, y, _) in enumerate(self.__entries, start=0):
            _, closest = self.__kdtree.query(np.column_stack([x, y]), 1)
            dist_x = self.__x[closest] - x
            dist_y = self.__y[closest] - y
            # Alpha
            ax_a = fig.add_subplot(nrows, 2, 2 * i + 1)
            ax_a.set_title(f'$\\Delta x$ for {label}')
            _, bins, _ = ax_a.hist(dist_x, bins=bins)
            # Delta
            ax_d = fig.add_subplot(nrows, 2, 2 * (i + 1))
            ax_d.set_title(f'$\\Delta y$ for {label}')
            _, bins, _ = ax_d.hist(dist_y, bins=bins)

        fig.tight_layout()
        return fig

    def _plot_dist_vs_mag(self):
        fig = plt.figure(figsize=_page_size)

        ax_a = fig.add_subplot(2, 1, 1)
        ax_a.set_title('$\\Delta x$ vs magnitude')
        ax_a.set_xlabel('Magnitude')
        ax_a.set_ylabel('$\\Delta x$')
        ax_a.grid(True)

        ax_d = fig.add_subplot(2, 1, 2, sharex=ax_a)
        ax_d.set_title('$\\Delta y$ vs magnitude')
        ax_d.set_xlabel('Magnitude')
        ax_d.set_ylabel('$\\Delta y$')
        ax_d.grid(True)

        max_dist = 0

        for i, (label, x, y, marker) in enumerate(self.__entries, start=0):
            _, closest = self.__kdtree.query(np.column_stack([x, y]), 1)
            dist_x = self.__x[closest] - x
            dist_y = self.__y[closest] - y

            max_dist = np.max([max_dist, np.abs(dist_x).max(), np.abs(dist_y).max()])

            ax_a.scatter(self.__mag[closest], dist_x, label=label, marker=marker)
            ax_d.scatter(self.__mag[closest], dist_y, label=label, marker=marker)

        ax_a.set_ylim(max_dist / 2, -max_dist / 2)
        ax_d.set_ylim(max_dist / 2, -max_dist / 2)
        ax_a.legend()
        fig.tight_layout()
        return fig

    def get_figures(self):
        return [self._plot_distances(), self._plot_dist_vs_mag()]


class Magnitude(Plot):
    def __init__(self, name, simulation):
        super(Magnitude, self).__init__()
        self.__mags = np.append(simulation[0].mag, simulation[1].mag)
        self.__kdtree = simulation[2]
        self.__figure = plt.figure(figsize=_page_size)
        self.__figure.subplots_adjust(left=0.07, right=0.93, hspace=0.0, wspace=0.2)

        gridspec = GridSpec(4, 1)

        # First plot: computed (Y) vs real (X)
        self.__ax_mag = self.__figure.add_subplot(gridspec.new_subplotspec((0, 0), 2))
        self.__ax_mag.set_title(f'Magnitude comparison for {name}')
        self.__ax_mag.set_ylabel('Measured value')
        self.__ax_mag.grid(True, linestyle=':')

        # Second plot: delta between computed and real
        self.__ax_mag.spines['bottom'].set_linestyle('--')
        self.__ax_delta = self.__figure.add_subplot(gridspec.new_subplotspec((2, 0), 1), sharex=self.__ax_mag)
        self.__ax_delta.set_facecolor('whitesmoke')
        self.__ax_delta.spines['top'].set_visible(False)
        self.__ax_delta.set_ylabel('$\Delta$')
        self.__ax_delta.axhline(0, color='gray')
        self.__ax_delta.grid(True, linestyle=':')
        self.__ax_delta.set_ylim(-1.1, 1.1)

        # Third plot: computed error
        self.__ax_err = self.__figure.add_subplot(gridspec.new_subplotspec((3, 0), 1), sharex=self.__ax_mag)
        self.__ax_err.set_facecolor('oldlace')
        self.__ax_err.set_ylabel('Measured error')
        self.__ax_err.set_xlabel('Real magnitude')
        self.__ax_err.grid(True, linestyle=':')

    def add(self, label, catalog, alpha_col, delta_col, mag_col, mag_err_col, marker=None):
        closest = stuff.get_closest(catalog, self.__kdtree, alpha_col, delta_col)
        source_mag = self.__mags[closest['source']]
        mag = get_column(catalog, mag_col)
        mag_err = get_column(catalog, mag_err_col)
        delta_mag = mag - source_mag
        self.__ax_mag.scatter(source_mag, mag, label=label, marker=marker)
        delta_col = self.__ax_delta.scatter(source_mag, delta_mag, label=label, marker=marker)
        self.__ax_err.scatter(source_mag, mag_err, marker=marker)
        # Mark there are some outside of the plot
        delta_above = source_mag[delta_mag > 1.]
        delta_below = source_mag[delta_mag < -1.]
        self.__ax_delta.scatter(delta_above, np.ones(delta_above.shape), marker='^', c=delta_col.get_facecolor())
        self.__ax_delta.scatter(delta_below, -np.ones(delta_below.shape), marker='v', c=delta_col.get_facecolor())

    def get_figures(self):
        self.__ax_mag.legend()
        return [self.__figure]


class Scatter(Plot):
    def __init__(self, name, simulation):
        super(Scatter, self).__init__()
        self.__mags = np.append(simulation[0].mag, simulation[1].mag)
        self.__kdtree = simulation[2]
        self.__figure = plt.figure(figsize=_page_size)
        self.__figure.subplots_adjust(left=0.07, right=0.93, hspace=0.0, wspace=0.2)

        gridspec = GridSpec(2, 1)

        # First plot: values
        self.__ax_val = self.__figure.add_subplot(gridspec.new_subplotspec((0, 0)))
        self.__ax_val.set_title(f'{name}')
        self.__ax_val.set_ylabel('Measured value')
        self.__ax_val.grid(True, linestyle=':')

        # Second plot: errors
        self.__ax_err = self.__figure.add_subplot(gridspec.new_subplotspec((1, 0)), sharex=self.__ax_val)
        self.__ax_err.set_facecolor('oldlace')
        self.__ax_err.set_ylabel('Measured error')
        self.__ax_err.set_xlabel('Real magnitude')
        self.__ax_err.grid(True, linestyle=':')

    def add(self, label, catalog, alpha_col, delta_col, val_col, err_col, marker=None):
        closest = stuff.get_closest(catalog, self.__kdtree, alpha_col, delta_col)
        source_mag = self.__mags[closest['source']]
        val = get_column(catalog, val_col)
        err = get_column(catalog, err_col)
        self.__ax_val.scatter(source_mag, val, label=label, marker=marker)
        self.__ax_err.scatter(source_mag, err, marker=marker)

    def get_figures(self):
        self.__ax_val.legend()
        return [self.__figure]


class Flags(Plot):
    def __init__(self, image):
        super(Flags, self).__init__()
        hdu = fits.open(image)[0]
        self.__image = hdu.data
        self.__wcs = WCS(hdu.header)
        self.__figure = plt.figure(figsize=_page_size)
        self.__ax1 = self.__figure.add_subplot(1, 2, 1, projection=self.__wcs)
        self.__ax1.imshow(self.__image, cmap=_img_cmap, norm=deepcopy(_img_norm))
        self.__ax2 = self.__figure.add_subplot(1, 2, 2, projection=self.__wcs)
        self.__ax2.imshow(self.__image, cmap=_img_cmap, norm=deepcopy(_img_norm))

    def __set(self, ax, label, catalog, alpha_col, delta_col, flag_col, is_sex2=False):
        ax.set_title(label)
        pix_coord = self.__wcs.all_world2pix(catalog[alpha_col], catalog[delta_col], 0)
        if is_sex2:
            flag_enum = stuff.Sex2SourceFlags
            flag_style = _sex2_flag_style
        else:
            flag_enum = stuff.SourceFlags
            flag_style = _flag_style

        for flag in flag_enum:
            if int(flag) == 0:
                continue
            flag_filter = (get_column(catalog, flag_col) & int(flag)).astype(np.bool)
            flag_color, flag_marker = flag_style[flag]
            if flag_filter.any():
                ax.scatter(
                    pix_coord[0][flag_filter], pix_coord[1][flag_filter],
                    c=flag_color, marker=flag_marker, label=flag
                )
        ax.legend()

    def set1(self, *args, **kwargs):
        self.__set(self.__ax1, *args, **kwargs)

    def set2(self, *args, **kwargs):
        self.__set(self.__ax2, *args, **kwargs)

    def get_figures(self):
        return [self.__figure]


class CumulativeHistogram(Plot):
    def __init__(self, simulation, nbins=50):
        super(CumulativeHistogram, self).__init__()
        self.__stars = simulation[0]
        self.__galaxies = simulation[1]
        self.__kdtree = simulation[2]
        self.__star_dists = []
        self.__galaxy_dists = []
        self.__figure = plt.Figure(figsize=_page_size)

        self.__ax_stars = self.__figure.add_subplot(2, 1, 1)
        self.__ax_stars.grid(True)
        self.__ax_stars.set_title('Stars CDF')
        self.__ax_stars.set_xlabel('Magnitude')

        _, self.__bins, _ = self.__ax_stars.hist(
            self.__stars.mag, bins=nbins,
            density=True, cumulative=True,
            label='Truth'
        )

        self.__ax_galaxies = self.__figure.add_subplot(2, 1, 2, sharex=self.__ax_stars)
        self.__ax_galaxies.grid(True)
        self.__ax_galaxies.set_title('Galaxies CDF')
        self.__ax_galaxies.set_xlabel('Magnitude')

        self.__ax_galaxies.hist(
            self.__galaxies.mag, bins=self.__bins,
            density=True, cumulative=True,
            label='Truth'
        )

    def add(self, label, catalog, alpha_col, delta_col, mag_col):
        closest = stuff.get_closest(catalog, self.__kdtree, alpha_col, delta_col)
        star_filter = (closest['source'] < len(self.__stars))
        galaxy_filter = (closest['source'] >= len(self.__galaxies))
        mag = get_column(catalog, mag_col)

        self.__ax_stars.hist(
            mag[closest['catalog'][star_filter]],
            density=True, cumulative=True,
            bins=self.__bins, histtype='step',
            label=label
        )
        self.__ax_galaxies.hist(
            mag[closest['catalog'][galaxy_filter]],
            density=True, cumulative=True,
            bins=self.__bins, histtype='step',
            label=label
        )

        self.__star_dists.append((label, mag[closest['catalog'][star_filter]]))
        self.__galaxy_dists.append((label, mag[closest['catalog'][galaxy_filter]]))

    def get_figures(self):
        text = []
        for a, b in zip(self.__star_dists, self.__star_dists[1:]):
            a_label, a_stars = a
            b_label, b_stars = b
            try:
                ks = ks_2samp(a_stars, b_stars)
                text.append(f'$H_0$({a_label} $\\approx$ {b_label}) p-value = {ks.pvalue:.2e}')
            except:
                pass
        self.__ax_stars.text(self.__stars.mag.min(), 0.1, '\n'.join(text), bbox=dict(facecolor='whitesmoke'))
        self.__ax_stars.legend()

        text = []
        for a, b in zip(self.__galaxy_dists, self.__galaxy_dists[1:]):
            a_label, a_galaxies = a
            b_label, b_galaxies = b
            try:
                ks = ks_2samp(a_galaxies, b_galaxies)
                text.append(f'$H_0$({a_label} $\\approx$ {b_label}) p-value = {ks.pvalue:.2e}')
            except:
                pass
        self.__ax_galaxies.text(self.__stars.mag.min(), 0.1, '\n'.join(text), bbox=dict(facecolor='whitesmoke'))

        self.__ax_galaxies.legend()
        self.__figure.tight_layout()
        return [self.__figure]


class Completeness(Plot):

    def __init__(self, image, simulation, max_dist=0.5):
        super(Completeness, self).__init__()
        hdu = fits.open(image)[0]
        self.__width = hdu.data.shape[0]
        self.__height = hdu.data.shape[1]
        self.__wcs = WCS(hdu.header)
        self.__max_dist = max_dist

        stars_x, stars_y, stars_mag = _get_sources_within_image(
            hdu.data, self.__wcs, simulation[0].ra, simulation[0].dec, simulation[0].mag
        )
        galx_x, galx_y, galx_mag = _get_sources_within_image(
            hdu.data, self.__wcs, simulation[1].ra, simulation[1].dec, simulation[1].mag
        )

        self.__stars = stars_mag
        self.__galaxies = galx_mag
        self.__all_mag = np.append(stars_mag, galx_mag)

        all_x = np.append(stars_x, galx_x)
        all_y = np.append(stars_y, galx_y)

        self.__kdtree = KDTree(np.column_stack([all_x, all_y]))

        min_mag = np.floor(np.min(np.append(self.__stars, self.__galaxies)))
        max_mag = np.ceil(np.max(np.append(self.__stars, self.__galaxies)))
        self.__edges = np.arange(min_mag - 0.5, max_mag + 0.5, 1)

        self.__stars_bins, _ = np.histogram(self.__stars, bins=self.__edges)
        self.__galaxies_bins, _ = np.histogram(self.__galaxies, bins=self.__edges)

        self.__star_recall = []
        self.__galaxy_recall = []
        self.__bad_detection = []
        self.__false_detection_nearest = []

    def add(self, label, catalog, x_col, y_col, mag_col):
        d, i = self.__kdtree.query(
            np.column_stack([catalog[x_col], catalog[y_col]])
        )
        real_found, real_counts = np.unique(i[d <= self.__max_dist], return_counts=True)
        # Found contains now the index of the "real" stars and galaxies with at least one match
        # If the index is < len(self.__stars), it is a star
        stars_found = real_found[real_found < len(self.__stars)]
        stars_hist, _ = np.histogram(self.__stars[stars_found], bins=self.__edges)
        stars_recall = stars_hist / self.__stars_bins
        # If the index is > len(self.__stars, it is a galaxy)
        galaxies_found = real_found[real_found >= len(self.__stars)] - len(self.__stars)
        galaxies_hist, _ = np.histogram(self.__galaxies[galaxies_found], bins=self.__edges)
        galaxies_recall = galaxies_hist / self.__galaxies_bins
        # Store recalls
        self.__star_recall.append((label, stars_recall))
        self.__galaxy_recall.append((label, galaxies_recall))

        # Last, detections that are too far from any "real" source
        # We show them binned by measured, and by nearest
        bad_filter = (d >= self.__max_dist)
        bad_mag = catalog[mag_col][bad_filter]
        bad_hist, _ = np.histogram(bad_mag, bins=self.__edges)
        self.__bad_detection.append((label, bad_hist / (galaxies_hist + stars_hist + bad_hist)))

    def __plot_recall(self, ax, edges, recall_list):
        ax.set_ylim(0., 100.)
        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        for label, recall in recall_list:
            ax.bar(bin_centers, recall * 100, alpha=0.5, label=label)
        ax.set_ylabel('%')
        ax.legend()
        ax.grid(True)

    def __plot_false(self, ax, edges, false_list):
        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        for label, count in false_list:
            ax.bar(bin_centers, count, alpha=0.5, label=label)
        ax.legend()
        ax.grid(True)

    def get_figures(self):
        fig_recall = plt.figure(figsize=_page_size)

        ax_stars = fig_recall.add_subplot(3, 1, 1)
        ax_stars.set_title(f'Star recall ($\\Delta < {self.__max_dist}$)')
        self.__plot_recall(ax_stars, self.__edges, self.__star_recall)

        ax_galaxies = fig_recall.add_subplot(3, 1, 2, sharex=ax_stars)
        ax_galaxies.set_title(f'Galaxy recall ($\\Delta < {self.__max_dist}$)')
        self.__plot_recall(ax_galaxies, self.__edges, self.__galaxy_recall)

        ax_bad_measured = fig_recall.add_subplot(3, 1, 3, sharex=ax_stars)
        ax_bad_measured.set_title(f'Percent of detections at $\\Delta \\geq {self.__max_dist}$, binned by measured magnitude')
        self.__plot_recall(ax_bad_measured, self.__edges, self.__bad_detection)

        fig_recall.tight_layout()
        return [fig_recall]


def generate_report(output, simulation, image, target, reference):
    with Report(output) as report:
        loc_map = Location(image, simulation)
        loc_map.add('SExtractor2', reference, 'X_IMAGE', 'Y_IMAGE', marker='1')
        loc_map.add('SExtractor++', target, 'pixel_centroid_x', 'pixel_centroid_y', marker='2')
        report.add(loc_map)

        dist = Distances(image, simulation)
        dist.add('SExtractor2', reference, 'X_IMAGE', 'Y_IMAGE', marker='o')
        dist.add('SExtractor++', target, 'pixel_centroid_x', 'pixel_centroid_y', marker='.')
        report.add(dist)

        completeness = Completeness(image, simulation)
        completeness.add('SExtractor2', reference, 'X_IMAGE', 'Y_IMAGE', 'MAG_ISO')
        completeness.add('SExtractor++', target, 'pixel_centroid_x', 'pixel_centroid_y', 'isophotal_mag')
        report.add(completeness)

        flux_iso = Scatter('FLUX_ISO vs isophotal_flux', simulation)
        flux_iso.add(
            'SExtractor2', reference,
            'ALPHA_SKY', 'DELTA_SKY',
            'FLUX_ISO', 'FLUXERR_ISO',
            marker='o'
        )
        flux_iso.add(
            'SExtractor++', target,
            'world_centroid_alpha', 'world_centroid_delta',
            'isophotal_flux', 'isophotal_flux_err',
            marker='.'
        )
        report.add(flux_iso)

        flux_auto = Scatter('FLUX_AUTO vs auto_flux', simulation)
        flux_auto.add(
            'SExtractor2', reference,
            'ALPHA_SKY', 'DELTA_SKY',
            'FLUX_AUTO', 'FLUXERR_AUTO',
            marker='o'
        )
        flux_auto.add(
            'SExtractor++', target,
            'world_centroid_alpha', 'world_centroid_delta',
            'auto_flux', 'auto_flux_err',
            marker='.'
        )
        report.add(flux_auto)

        mag_iso = Magnitude('MAG_ISO vs isophotal_mag', simulation)
        mag_iso.add(
            'SExtractor2', reference,
            'ALPHA_SKY', 'DELTA_SKY',
            'MAG_ISO', 'MAGERR_ISO',
            marker='o'
        )
        mag_iso.add(
            'SExtractor++', target,
            'world_centroid_alpha', 'world_centroid_delta',
            'isophotal_mag', 'isophotal_mag_err',
            marker='.'
        )
        report.add(mag_iso)

        iso_hist = CumulativeHistogram(simulation)
        iso_hist.add(
            'SExtractor2 MAG_ISO', reference,
            'ALPHA_SKY', 'DELTA_SKY',
            'MAG_ISO'
        )
        iso_hist.add(
            'SExtractor++ isophotal_mag', target,
            'world_centroid_alpha', 'world_centroid_delta',
            'isophotal_mag'
        )
        report.add(iso_hist)

        mag_auto = Magnitude('MAG_AUTO vs auto_mag', simulation)
        mag_auto.add(
            'SExtractor2', reference,
            'ALPHA_SKY', 'DELTA_SKY',
            'MAG_AUTO', 'MAGERR_AUTO',
            marker='o'
        )
        mag_auto.add(
            'SExtractor++', target,
            'world_centroid_alpha', 'world_centroid_delta',
            'auto_mag', 'auto_mag_err',
            marker='.'
        )
        report.add(mag_auto)

        auto_hist = CumulativeHistogram(simulation)
        auto_hist.add(
            'SExtractor2 MAG_AUTO', reference,
            'ALPHA_SKY', 'DELTA_SKY',
            'MAG_AUTO'
        )
        auto_hist.add(
            'SExtractor++ auto_mag', target,
            'world_centroid_alpha', 'world_centroid_delta',
            'auto_mag'
        )
        report.add(auto_hist)

        # Try apertures columns
        if 'aperture_mag' in target.dtype.names:
            target_aperture_mag = target['aperture_mag']
            aper_hist = CumulativeHistogram(simulation)
            n_aper = target_aperture_mag.shape[2]
            for i in range(n_aper):
                aper_hist.add(
                    f'SExtractor2 aperture {i}', reference,
                    'ALPHA_SKY', 'DELTA_SKY',
                    f'MAG_APER:{i}'
                )
                aper_hist.add(
                    f'SExtractor++ aperture {i}', target,
                    'world_centroid_alpha', 'world_centroid_delta',
                    f'aperture_mag:0:{i}'
                )
            report.add(aper_hist)

        src_flags = Flags(image)
        src_flags.set1(
            'SExtractor2', reference,
            'ALPHA_SKY', 'DELTA_SKY', 'FLAGS'
        )
        src_flags.set2(
            'SExtractor++ source_flags', target,
            'world_centroid_alpha', 'world_centroid_delta', 'source_flags'
        )
        report.add(src_flags)

        auto_flags = Flags(image)
        auto_flags.set1(
            'SExtractor2', reference,
            'ALPHA_SKY', 'DELTA_SKY', 'FLAGS'
        )
        auto_flags.set2(
            'SExtractor++ auto_flags', target,
            'world_centroid_alpha', 'world_centroid_delta', 'auto_flags'
        )
        report.add(auto_flags)
