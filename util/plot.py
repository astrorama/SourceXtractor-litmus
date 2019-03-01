import abc

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
from scipy.stats import ks_2samp

from util import stuff
from . import get_column

# A4 size in inches
_page_size = (11.7, 8.3)

# Color map used for the images
_img_cmap = plt.get_cmap('gist_gray')

# SExtractor++ flag style. We define them here instead of relying on the auto-style,
# so they stays consistent between runs, images, etc.
_flag_style = {
    stuff.SourceFlags.BIASED: ('red', '1'),
    stuff.SourceFlags.BLENDED: ('blue', '2'),
    stuff.SourceFlags.SATURATED: ('orange', '+'),
    stuff.SourceFlags.BOUNDARY: ('pink', '3'),
    stuff.SourceFlags.NEIGHBORS: ('cyan', '4'),
    stuff.SourceFlags.OUTSIDE: ('white', 'x')
}
# SExtractor2 flag style. We define them here instead of relying on the auto-style,
# so they stays consistent between runs, images, etc.
# Also, we try to match to the SExtractor++ style when there is a correspondence, so
# visually comparing is easier
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


class Image(object):
    """
    Contains a FITS image, and its World Coordinate System, and an optional,
    associated, weight image.
    """

    def __init__(self, image, weight_image=None, hdu_index=0):
        """
        Constructor.
        :param image:
            Path to the FITS image.
        :param weight_image:
            Path to the FITS weight image. It can be None.
        :param hdu_index:
            Index of the HDU that contains the image data. Defaults to 0 (the primary one).
        """
        self.__image = fits.open(image)[hdu_index]
        self.__weight = fits.open(weight_image)[hdu_index] if weight_image else None
        self.__wcs = WCS(self.__image.header)

    @property
    def wcs(self):
        """
        :return:
            The World Coordinate System of the image.
        """
        return self.__wcs

    @property
    def data(self):
        """
        :return:
            numpy array with the data from the image data.
        """
        return self.__image.data

    @property
    def weight(self):
        """
        :return:
            numpy array with the data form the weight image, or None if there is no weight image.
        """
        return self.__weight.data if self.__weight is not None else None

    @property
    def size(self):
        """
        :return:
            Tuple with the size of the image data.
        """
        return self.__image.shape

    def for_display(self):
        """
        :return:
            Image data, massaged so it can be plotted nicely with matplotlib.
        """
        return ZScaleInterval()(self.data, clip=True)

    def get_contained_sources(self, ra, dec, *args):
        """
        Filter sources that are outside of the image:
            1. It projects the right ascension and declination to pixel coordinates.
            2. Remove sources that are projected outside of the image.
            3. If there is a weight image, remove sources that correspond to areas with a weight of 0.
               (Useful for coadded images, for instance)
        :param ra:
            Right ascension
        :param dec:
            Declination
        :param args:
            Arbitrary number of parameters, where each one is expected to be a numpy array with the same
            size of ra and dec. There parameters will also be filtered out.
        :return:
            A tuple with the x and y pixel coordinates for the sources that are within the image,
            and one extra element corresponding to each extra argument, also filtered.
        """
        assert len(ra) == len(dec)
        for a in args:
            assert len(a) == len(ra)

        h, w = self.size
        pix_coords = self.__wcs.all_world2pix(ra, dec, 1)
        pix_x = pix_coords[0]
        pix_y = pix_coords[1]
        inside_image = (pix_x >= 0) & (pix_x < w) & (pix_y >= 0) & (pix_y < h)
        # If we have a weight map, filter out those with weight 0
        if self.__weight is not None:
            weight_filter = self.weight[pix_y[inside_image].astype(np.int), pix_x[inside_image].astype(np.int)] != 0.
            inside_image[inside_image] = weight_filter
        pix_x = pix_coords[0][inside_image]
        pix_y = pix_coords[1][inside_image]
        return tuple([pix_x, pix_y] + [a[inside_image] for a in args])


class Plot(object):
    """
    Base class for plots embeddable on a report.
    """

    @abc.abstractmethod
    def get_figures(self):
        """
        An implementation must return here the list of the figures to be added to the report.
        """
        pass


class Report(object):
    """
    A report is a collection of figures. Each one will be written to a separate PDF page.
    """

    def __init__(self, path):
        """
        :param path:
            Path where to write the report.
        """
        self.__pdf = PdfPages(path)

    def __enter__(self):
        """
        Allows to use this class inside of a `with ... as` block.
        :return:
            self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when leaving a `with ... as` block. Writes the report.
        """
        self.close()

    def close(self):
        """
        Write the report.
        :return:
        """
        self.__pdf.close()

    def add(self, plot):
        """
        Add a plot to the report.
        :param plot:
            A Plot instance
        """
        assert isinstance(plot, Plot)
        for fig in plot.get_figures():
            self.__pdf.savefig(fig)
            plt.close(fig)


class Location(Plot):
    """
    Display the image, with the detections overlaid.
    """

    def __init__(self, image, simulation):
        """
        Constructor.
        :param image:
            An instance of Image.
        :param simulation:
            The simulation that originated the image.
        """
        super(Location, self).__init__()
        self.__figure = plt.figure(figsize=_page_size)
        self.__ax = self.__figure.add_subplot(1, 1, 1)
        self.__ax.set_title('Location')
        self.__ax.imshow(image.for_display(), cmap=_img_cmap)

    def add(self, label, catalog, x_col, y_col, size_col=None, **kwargs):
        """
        Add a new detection.
        :param label:
            Label for the detection instance.
        :param catalog:
            Astropy table with the data.
        :param x_col:
            Column name for the X coordinate.
        :param y_col:
            Column name for the Y coordinate.
        :param size_col:
            Column name for the size column, used to scale the size of the marker.
        :param kwargs:
            Passed down to the scatter method from matplotlib
        """
        self.__ax.scatter(
            catalog[x_col], catalog[y_col],
            label=label, s=np.sqrt(catalog[size_col]) * 3 if size_col else None,
            **kwargs
        )

    def get_figures(self):
        """
        :return: The list of generated figures
        """
        self.__ax.legend()
        self.__figure.tight_layout()
        return [self.__figure]


class Distances(Plot):
    """
    Display two figures:
        1. A histogram with the distances for both X and Y coordinates.
        2. A scatter plot with the distances for X and Y coordinates, as a funciton of the real magnitude.
    """

    def __init__(self, image, simulation):
        """
        Constructor.
        :param image:
            An instance of Image, used to project the simulation into X and Y coordinates.
        :param simulation:
            The simulation that originated the image.
        """
        super(Distances, self).__init__()
        stars_x, stars_y, stars_mag = image.get_contained_sources(
            simulation[0].ra, simulation[0].dec, simulation[0].mag
        )
        galaxies_x, galaxies_y, galaxies_mag = image.get_contained_sources(
            simulation[1].ra, simulation[1].dec, simulation[1].mag
        )

        self.__x = np.append(stars_x, galaxies_x)
        self.__y = np.append(stars_y, galaxies_y)
        self.__mag = np.append(stars_mag, galaxies_mag)
        self.__kdtree = KDTree(np.column_stack([self.__x, self.__y]))
        self.__entries = []

    def add(self, label, catalog, x_col, y_col, **kwargs):
        """
        Add a new detection.
        :param label:
            Label for the detection instance.
        :param catalog:
            Astropy table with the data.
        :param x_col:
            Column name for the X coordinate.
        :param y_col:
            Column name for the Y coordinate.
        :param kwargs:
            Passed down to the scatter method from matplotlib.
        """
        self.__entries.append((label, catalog[x_col], catalog[y_col], kwargs))

    def _plot_distances(self):
        """
        Draw the histogram with the distances.
        """
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
        """
        Draw the scatter plot with the distance per axis vs magnitude.
        """
        fig = plt.figure(figsize=_page_size)

        ax_x = fig.add_subplot(2, 1, 1)
        ax_x.set_title('$\\Delta x$ vs magnitude')
        ax_x.set_xlabel('Magnitude')
        ax_x.set_ylabel('$\\Delta x$')
        ax_x.grid(True)

        ax_y = fig.add_subplot(2, 1, 2, sharex=ax_x)
        ax_y.set_title('$\\Delta y$ vs magnitude')
        ax_y.set_xlabel('Magnitude')
        ax_y.set_ylabel('$\\Delta y$')
        ax_y.grid(True)

        max_dist = 0

        for i, (label, x, y, kwargs) in enumerate(self.__entries, start=0):
            _, closest = self.__kdtree.query(np.column_stack([x, y]), 1)
            dist_x = self.__x[closest] - x
            dist_y = self.__y[closest] - y

            max_dist = np.max([max_dist, np.abs(dist_x).max(), np.abs(dist_y).max()])

            ax_x.scatter(self.__mag[closest], dist_x, label=label, **kwargs)
            ax_y.scatter(self.__mag[closest], dist_y, label=label, **kwargs)

        # Limit the display to the half the maximum distance, or 5 if it is bigger
        max_dist = np.min([max_dist / 2, 5])
        ax_x.set_ylim(max_dist, -max_dist)
        ax_y.set_ylim(max_dist, -max_dist)

        ax_x.legend()
        fig.tight_layout()
        return fig

    def get_figures(self):
        return [self._plot_distances(), self._plot_dist_vs_mag()]


class Magnitude(Plot):
    """
    Display three scatter plots as a single plot:
        1. Measured magnitude as a function of real magnitude (that of the closest source).
        2. The delta between measured and real.
        3. The error as measured by the detection software.
    """

    def __init__(self, name, simulation):
        """
        Constructor.
        :param name:
            An identifying name for the plot (i.e. MAG_ISO vs isophotal_mag).
        :param simulation:
            The simulation that originated the image.
        """
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

    def add(self, label, catalog, alpha_col, delta_col, mag_col, mag_err_col, **kwargs):
        """
        :param label:
            Label for the detection instance.
        :param catalog:
            Astropy table with the data.
        :param alpha_col:
            Column name for the Right Ascension world coordinate.
        :param delta_col:
            Column name for the Declination world coordinate.
        :param mag_col:
            Column name for the magnitude measurement.
        :param mag_err_col:
            Column name for the error of the magnitude measurement.
        :param kwargs:
            Passed down to the scatter method from matplotlib.
        """
        # Get the magnitude for the closest true source
        closest = stuff.get_closest(catalog, self.__kdtree, alpha_col, delta_col)
        source_mag = self.__mags[closest['source']]

        # Get measured magnitude, error, and calculate the delta to the true one
        mag = get_column(catalog, mag_col)
        mag_err = get_column(catalog, mag_err_col)
        delta_mag = mag - source_mag

        # Plot
        self.__ax_mag.scatter(source_mag, mag, label=label, **kwargs)
        delta_col = self.__ax_delta.scatter(source_mag, delta_mag, label=label, **kwargs)
        self.__ax_err.scatter(source_mag, mag_err, **kwargs)

        # Mark there are some outside of the plot
        delta_above = source_mag[delta_mag > 1.]
        delta_below = source_mag[delta_mag < -1.]
        self.__ax_delta.scatter(delta_above, np.ones(delta_above.shape), marker='^', c=delta_col.get_facecolor())
        self.__ax_delta.scatter(delta_below, -np.ones(delta_below.shape), marker='v', c=delta_col.get_facecolor())

    def get_figures(self):
        """
        :return: The list of generated figures
        """
        self.__ax_mag.legend()
        return [self.__figure]


class Scatter(Plot):
    """
    Display two scatter plots as a single one:
        1. Whatever column value as a function of real magnitude
        2. The error of that value as a function of real magnitude
    """

    def __init__(self, name, simulation):
        """
        :param name:
            An identifying name for the plot (i.e. FLUX_ISO vs isophotal_flux).
        :param simulation:
            The simulation that originated the image.
        """
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

    def add(self, label, catalog, alpha_col, delta_col, val_col, err_col, **kwargs):
        """
        :param label:
            Label for the detection instance.
        :param catalog:
            Astropy table with the data.
        :param alpha_col:
            Column name for the Right Ascension world coordinate.
        :param delta_col:
            Column name for the Declination world coordinate.
        :param val_col:
            Column name for the magnitude measurement.
        :param err_col:
            Column name for the error of the magnitude measurement.
        :param kwargs:
            Passed down to the scatter method from matplotlib.
        """
        closest = stuff.get_closest(catalog, self.__kdtree, alpha_col, delta_col)
        source_mag = self.__mags[closest['source']]
        val = get_column(catalog, val_col)
        err = get_column(catalog, err_col)
        self.__ax_val.scatter(source_mag, val, label=label, **kwargs)
        self.__ax_err.scatter(source_mag, err, **kwargs)

    def get_figures(self):
        """
        :return: The list of generated figures
        """
        self.__ax_val.legend()
        return [self.__figure]


class Flags(Plot):
    """
    Display the image, with the flagged sources overlaid.
    """

    def __init__(self, image):
        """
        :param image:
            An instance of Image.
        """
        super(Flags, self).__init__()
        self.__figure = plt.figure(figsize=_page_size)
        self.__ax1 = self.__figure.add_subplot(1, 2, 1)
        self.__ax1.imshow(image.for_display(), cmap=_img_cmap)
        self.__ax2 = self.__figure.add_subplot(1, 2, 2)
        self.__ax2.imshow(image.for_display(), cmap=_img_cmap)

    @staticmethod
    def __set(ax, label, catalog, x_col, y_col, flag_col, is_sex2 = False):
        """
        Convenience method to plot the flags over a matplotlib ax.
        :param ax:
            ax where to plot the flags.
        :param label:
            Label for the detection instance.
        :param catalog:
            Astropy table with the data.
        :param x_col:
            Column name for the X coordinate.
        :param y_col:
            Column name for the Y coordinate.
        :param flag_col:
            Column name for the flags.
        :param is_sex2:
            True to use SExtractor2 flags, False to use SExtractor++ flags instead.
        """
        ax.set_title(label)
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
                    catalog[x_col][flag_filter], catalog[y_col][flag_filter],
                    c=flag_color, marker=flag_marker, label=flag
                )
        ax.legend()

    def set_sextractor2(self, *args, **kwargs):
        """
        Set the flags from the SExtractor2 run.
        Forward the parameters to __set
        """
        kwargs['is_sex2'] = True
        self.__set(self.__ax1, *args, **kwargs)

    def set_sextractorpp(self, *args, **kwargs):
        """
        Set the flags from the SExtractor++ run.
        Forward the parameters to __set
        """
        self.__set(self.__ax2, *args, **kwargs)

    def get_figures(self):
        """
        :return: The list of generated figures
        """
        return [self.__figure]


class Histogram(Plot):
    """
    Display the 'true' magnitude histogram, overlaid with measured magnitude histograms.
    """

    def __init__(self, image, simulation, nbins = 20):
        """
        :param image:
            An instance of Image, used to project the simulation into X and Y coordinates.
        :param simulation:
            The simulation that originated the image. It is used to split by star and galaxy.
        :param nbins:
            Number of bins for the histogram
        """
        super(Histogram, self).__init__()
        self.__stars = simulation[0]
        self.__galaxies = simulation[1]
        self.__kdtree = simulation[2]
        self.__star_dists = []
        self.__galaxy_dists = []
        self.__figure = plt.Figure(figsize=_page_size)

        _, _, stars_mag = image.get_contained_sources(
            simulation[0].ra, simulation[0].dec, simulation[0].mag
        )
        _, _, galx_mag = image.get_contained_sources(
            simulation[1].ra, simulation[1].dec, simulation[1].mag
        )

        self.__ax_stars = self.__figure.add_subplot(2, 1, 1)
        self.__ax_stars.yaxis.grid(True)
        self.__ax_stars.set_title('Stars')
        self.__ax_stars.set_xlabel('Magnitude')

        _, self.__bins, _ = self.__ax_stars.hist(
            stars_mag, bins=nbins, color='gray',
            label='Truth'
        )

        self.__ax_galaxies = self.__figure.add_subplot(2, 1, 2, sharex=self.__ax_stars)
        self.__ax_galaxies.yaxis.grid(True)
        self.__ax_galaxies.set_title('Galaxies')
        self.__ax_galaxies.set_xlabel('Magnitude')

        self.__ax_galaxies.hist(
            galx_mag, bins=self.__bins, color='gray',
            label='Truth'
        )

    def add(self, label, catalog, alpha_col, delta_col, mag_col):
        """
        :param label:
            Label for the detection instance.
        :param catalog:
            Astropy table with the data.
        :param alpha_col:
            Column name for the Right Ascension world coordinate.
        :param delta_col:
            Column name for the Declination world coordinate.
        :param mag_col:
            Column name for the magnitude measurement.
        """
        closest = stuff.get_closest(catalog, self.__kdtree, alpha_col, delta_col)
        star_filter = (closest['source'] < len(self.__stars))
        galaxy_filter = (closest['source'] >= len(self.__galaxies))
        mag = get_column(catalog, mag_col)

        self.__ax_stars.hist(
            mag[closest['catalog'][star_filter]],
            bins=self.__bins, histtype='step', lw=2,
            label=label
        )
        self.__ax_galaxies.hist(
            mag[closest['catalog'][galaxy_filter]],
            bins=self.__bins, histtype='step', lw=2,
            label=label
        )

        self.__star_dists.append((label, mag[closest['catalog'][star_filter]]))
        self.__galaxy_dists.append((label, mag[closest['catalog'][galaxy_filter]]))

    def get_figures(self):
        """
        :return: The list of generated figures
        """
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
    """
    Display three histograms, in percentages:
        1. Star recall: number of true stars found on the catalog, binned by the true magnitude.
        2. Galaxy recall: number of true galaxies found on the catalog, binned by the true magnitude.
        3. Detections done too far from any true sources, binned by the measured magnitude
    """

    def __init__(self, image, simulation, max_dist=0.5):
        """
        :param image:
            An instance of Image, used to project the simulation into X and Y coordinates.
        :param simulation:
            The simulation that originated the image.
        :param max_dist:
            Maximum distance, in pixels, to be a source considered a match to a real source.
        """
        super(Completeness, self).__init__()
        self.__max_dist = max_dist

        stars_x, stars_y, stars_mag = image.get_contained_sources(
            simulation[0].ra, simulation[0].dec, simulation[0].mag
        )
        galx_x, galx_y, galx_mag = image.get_contained_sources(
            simulation[1].ra, simulation[1].dec, simulation[1].mag
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
        """

        :param label:
            Label for the detection instance.
        :param catalog:
            Astropy table with the data.
        :param x_col:
            Column name for the X coordinate.
        :param y_col:
            Column name for the Y coordinate.
        :param mag_col:
            Column name for the magnitude measurement.
        """
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

    @staticmethod
    def __plot_recall(ax, edges, recall_list, real_counts):
        """
        Plot the star and galaxy recalls.
        """
        ax.set_ylim(0., 100.)
        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        bars = None
        for label, recall in recall_list:
            bars = ax.bar(bin_centers, recall * 100, alpha=0.5, label=label)
        if bars is not None and real_counts is not None:
            for b, r in zip(bars, real_counts):
                ax.text(b.get_x() + b.get_width() / 2.5, b.get_y() + 10, str(r))
        ax.set_ylabel('%')
        ax.legend()
        ax.yaxis.grid(True)

    @staticmethod
    def __plot_false(ax, edges, false_list):
        """
        Plot the 'false' (or rather, too far) detections.
        """
        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        for label, count in false_list:
            ax.bar(bin_centers, count, alpha=0.5, label=label)
        ax.legend()
        ax.yaxis.grid(True)

    def get_figures(self):
        """
        :return: The list of generated figures
        """
        fig_recall = plt.figure(figsize=_page_size)

        ax_stars = fig_recall.add_subplot(3, 1, 1)
        ax_stars.set_title(f'Star recall ($\\Delta < {self.__max_dist}$px)')
        self.__plot_recall(ax_stars, self.__edges, self.__star_recall, self.__stars_bins)

        ax_galaxies = fig_recall.add_subplot(3, 1, 2, sharex=ax_stars)
        ax_galaxies.set_title(f'Galaxy recall ($\\Delta < {self.__max_dist}$px)')
        self.__plot_recall(ax_galaxies, self.__edges, self.__galaxy_recall, self.__galaxies_bins)

        ax_bad_measured = fig_recall.add_subplot(3, 1, 3, sharex=ax_stars)
        ax_bad_measured.set_title(
            f'Percent of detections at $\\Delta \\geq {self.__max_dist}$px, binned by measured magnitude')
        self.__plot_recall(ax_bad_measured, self.__edges, self.__bad_detection, None)

        fig_recall.tight_layout()
        return [fig_recall]


def generate_report(output, simulation, image_path, target, reference, weight_image=None):
    """
    Convenience function to generate a typical report.
    :param output:
        Path for the output PDF.
    :param simulation:
        The original simulation.
    :param image_path:
        Path to the detection image.
    :param target:
        The target catalog, typically generated by SExtractor++.
    :param reference:
        The reference catalog, typically generated by SExtractor2.
    :param weight_image:
        An optional weight image, used to filter out sources from the simulation
    """

    with Report(output) as report:
        image = Image(image_path, weight_image=weight_image)

        loc_map = Location(image, simulation)
        loc_map.add('SExtractor2', reference, 'X_IMAGE', 'Y_IMAGE', 'ISOAREA_IMAGE', marker='o', facecolors='none',
                    edgecolors='red')
        loc_map.add('SExtractor++', target, 'pixel_centroid_x', 'pixel_centroid_y', 'area', marker='.',
                    facecolors='none', edgecolors='orange')
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

        iso_hist = Histogram(image, simulation)
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

        auto_hist = Histogram(image, simulation)
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
            aper_hist = Histogram(image, simulation)
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
        src_flags.set_sextractor2(
            'SExtractor2', reference,
            'X_IMAGE', 'Y_IMAGE', 'FLAGS'
        )
        src_flags.set_sextractorpp(
            'SExtractor++ source_flags', target,
            'pixel_centroid_x', 'pixel_centroid_y', 'source_flags'
        )
        report.add(src_flags)

        auto_flags = Flags(image)
        auto_flags.set_sextractor2(
            'SExtractor2', reference,
            'X_IMAGE', 'Y_IMAGE', 'FLAGS'
        )
        auto_flags.set_sextractorpp(
            'SExtractor++ auto_flags', target,
            'pixel_centroid_x', 'pixel_centroid_y', 'auto_flags'
        )
        report.add(auto_flags)
