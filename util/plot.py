import abc
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import colors

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp

from util import stuff
from . import get_column

_page_size = (11.7, 8.3)
_img_cmap = plt.get_cmap('Greys_r')
_img_norm = colors.SymLogNorm(100, linscale=2)

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
        w, h = hdu.data.shape
        self.__wcs = WCS(hdu.header)
        self.__figure = plt.figure(figsize=_page_size)
        self.__ax = self.__figure.add_subplot(1, 1, 1, projection=self.__wcs)
        self.__ax.set_title('Location')
        self.__ax.imshow(self.__image, cmap=_img_cmap, norm=_img_norm)
        # Stars
        mag_cmap = plt.get_cmap('magma_r')
        pix_coord = self.__wcs.all_world2pix(simulation[0].ra, simulation[0].dec, 0)
        pix_filter = (0 <= pix_coord[0]) & (pix_coord[0] < w) & (0 <= pix_coord[1]) & (pix_coord[1] < h)
        self.__ax.scatter(
            pix_coord[0][pix_filter], pix_coord[1][pix_filter],
            c=simulation[0].mag[pix_filter], marker='o',
            cmap=mag_cmap
        )
        pix_coord = self.__wcs.all_world2pix(simulation[1].ra, simulation[1].dec, 0)
        pix_filter = (0 <= pix_coord[0]) & (pix_coord[0] < w) & (0 <= pix_coord[1]) & (pix_coord[1] < h)
        cax = self.__ax.scatter(
            pix_coord[0][pix_filter], pix_coord[1][pix_filter],
            c=simulation[1].mag[pix_filter], marker='h',
            cmap=mag_cmap
        )
        cbar = self.__figure.colorbar(cax)
        cbar.ax.set_ylabel('Magnitude (truth)')

    def add(self, label, catalog, alpha, delta, marker=None):
        pix_coord = self.__wcs.all_world2pix(catalog[alpha], catalog[delta], 0)
        self.__ax.scatter(pix_coord[0], pix_coord[1], marker=marker, label=label, alpha=0.8,)

    def get_figures(self):
        self.__ax.legend()
        return [self.__figure]


class Distances(Plot):
    def __init__(self, simulation):
        super(Distances, self).__init__()
        self.__ra = np.append(simulation[0].ra, simulation[1].ra)
        self.__dec = np.append(simulation[0].dec, simulation[1].dec)
        self.__mag = np.append(simulation[0].mag, simulation[1].mag)
        self.__kdtree = simulation[2]
        self.__entries = []

    def add(self, label, catalog, alpha_col, delta_col, marker):
        self.__entries.append((label, catalog[alpha_col], catalog[delta_col], marker))

    def _plot_distances(self):
        fig = plt.figure(figsize=_page_size)
        nrows = len(self.__entries)
        bins = 20
        for i, (label, alpha, delta, _) in enumerate(self.__entries, start=0):
            _, closest = self.__kdtree.query(np.column_stack([alpha, delta]), 1)
            distances_alpha = np.abs(self.__ra[closest] - alpha)
            distances_delta = np.abs(self.__dec[closest] - delta)
            # Alpha
            ax_a = fig.add_subplot(nrows, 2, 2 * i + 1)
            ax_a.set_title(f'$\\Delta\\alpha$ for {label}')
            _, bins, _ = ax_a.hist(distances_alpha, bins=bins)
            # Delta
            ax_d = fig.add_subplot(nrows, 2, 2 * (i + 1))
            ax_d.set_title(f'$\\Delta\\delta$ for {label}')
            _, bins, _ = ax_d.hist(distances_delta, bins=bins)

        fig.tight_layout()
        return fig

    def _plot_dist_vs_mag(self):
        fig = plt.figure(figsize=_page_size)

        ax_a = fig.add_subplot(2, 1, 1)
        ax_a.set_title('$\\Delta\\alpha$ vs magnitude')
        ax_a.set_xlabel('Magnitude')
        ax_a.set_ylabel('$\\Delta\\alpha$')
        ax_a.grid(True)

        ax_d = fig.add_subplot(2, 1, 2, sharex=ax_a)
        ax_d.set_title('$\\Delta\\delta$ vs magnitude')
        ax_d.set_xlabel('Magnitude')
        ax_d.set_ylabel('$\\Delta\\delta$')
        ax_d.grid(True)

        max_dist = 0

        for i, (label, alpha, delta, marker) in enumerate(self.__entries, start=0):
            _, closest = self.__kdtree.query(np.column_stack([alpha, delta]), 1)
            distances_alpha = self.__ra[closest] - alpha
            distances_delta = self.__dec[closest] - delta

            max_dist = np.max([max_dist, np.abs(distances_alpha).max(), np.abs(distances_delta).max()])

            ax_a.scatter(self.__mag[closest], distances_alpha, label=label, marker=marker)
            ax_d.scatter(self.__mag[closest], distances_delta, label=label, marker=marker)

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
        self.__ax1.imshow(self.__image, cmap=_img_cmap, norm=_img_norm)
        self.__ax2 = self.__figure.add_subplot(1, 2, 2, projection=self.__wcs)
        self.__ax2.imshow(self.__image, cmap=_img_cmap, norm=_img_norm)

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


def generate_report(output, simulation, image, target, reference):
    with Report(output) as report:
        loc_map = Location(image, simulation)
        loc_map.add('SExtractor2', reference, 'ALPHA_SKY', 'DELTA_SKY', marker='1')
        loc_map.add('SExtractor++', target, 'world_centroid_alpha', 'world_centroid_delta', marker='2')
        report.add(loc_map)

        dist = Distances(simulation)
        dist.add('SExtractor2', reference, 'ALPHA_SKY', 'DELTA_SKY', marker='o')
        dist.add('SExtractor++', target, 'world_centroid_alpha', 'world_centroid_delta', marker='.')
        report.add(dist)

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
