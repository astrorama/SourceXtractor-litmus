import abc
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import colors

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from util import stuff
from . import get_column

_page_size = (11.7, 8.3)
_img_cmap = plt.get_cmap('Greys_r')
_img_norm = colors.SymLogNorm(10)

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
    def __init__(self, image):
        hdu = fits.open(image)[0]
        self.__image = hdu.data
        self.__wcs = WCS(hdu.header)
        self.__figure = plt.figure(figsize=_page_size)
        self.__ax = self.__figure.add_subplot(1, 1, 1, projection=self.__wcs)
        self.__ax.set_title('Location')
        self.__ax.imshow(self.__image, cmap=_img_cmap, norm=_img_norm)

    def add(self, label, catalog, alpha, delta, marker=None):
        pix_coord = self.__wcs.all_world2pix(catalog[alpha], catalog[delta], 0)
        self.__ax.scatter(pix_coord[0], pix_coord[1], marker=marker, label=label)

    def get_figures(self):
        self.__ax.legend()
        return [self.__figure]


class Distances(Plot):
    def __init__(self, simulation):
        self.__kdtree = simulation[2]
        self.__entries = []

    def add(self, label, catalog, alpha, delta):
        self.__entries.append((label, catalog[alpha], catalog[delta]))

    def get_figures(self):
        fig = plt.figure(figsize=_page_size)
        nrows = len(self.__entries)
        bins = 50
        for i, (label, alpha, delta) in enumerate(self.__entries, start=1):
            distances = []
            for a, d in zip(alpha, delta):
                d, _ = self.__kdtree.query([a, d], 1)
                distances.append(d)
            ax = fig.add_subplot(nrows, 1, i)
            ax.set_title(f'Distances for {label}')
            _, bins, _ = ax.hist(distances, bins=bins)
        fig.tight_layout()
        return [fig]


class Magnitude(Plot):
    def __init__(self, name, kdtree, real_values):
        self.__mags = real_values
        self.__kdtree = kdtree
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
        self.__ax_mag.scatter(source_mag, mag, label=label, marker=marker)
        self.__ax_delta.scatter(source_mag, mag - source_mag, label=label, marker=marker)
        self.__ax_err.scatter(source_mag, mag_err, marker=marker)

    def get_figures(self):
        self.__ax_mag.legend()
        return [self.__figure]


class Flags(Plot):
    def __init__(self, image):
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