from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from . import stuff, get_column

_default_target_columns = [
    [
        ('isophotal_flux', 'isophotal_flux_err'),
        ('auto_flux', 'auto_flux_err'),
        ('aperture_flux:0:0', 'aperture_flux_err:0:0'),
        ('aperture_flux:0:1', 'aperture_flux_err:0:1'),
        ('aperture_flux:0:2', 'aperture_flux_err:0:2'),
    ],
    [
        ('isophotal_mag', 'isophotal_mag_err'),
        ('auto_mag', 'auto_mag_err'),
        ('aperture_mag:0:0', 'aperture_mag_err:0:0'),
        ('aperture_mag:0:1', 'aperture_mag_err:0:1'),
        ('aperture_mag:0:2', 'aperture_mag_err:0:2')
    ]
]
_default_reference_columns = [
    [
        ('FLUX_ISO', 'FLUXERR_ISO'),
        ('FLUX_AUTO', 'FLUXERR_AUTO'),
        ('FLUX_APER:0', 'FLUXERR_APER:0'),
        ('FLUX_APER:1', 'FLUXERR_APER:1'),
        ('FLUX_APER:2', 'FLUXERR_APER:2'),
    ],
    [
        ('MAG_ISO', 'MAGERR_ISO'),
        ('MAG_AUTO', 'MAGERR_AUTO'),
        ('MAG_APER:0', 'MAGERR_APER:0'),
        ('MAG_APER:1', 'MAGERR_APER:1'),
        ('MAG_APER:2', 'MAGERR_APER:2')
    ]
]
_default_target_flag_columns = [
    'source_flags', 'auto_flags', 'aperture_flags:0:0', 'aperture_flags:0:1', 'aperture_flags:0:2'
]
_page_size = (11.7, 8.3)
_img_cmap = plt.get_cmap('Greys_r')
_img_norm = colors.SymLogNorm(10)

_ref_label = 'SExtractor 2'
_target_label = 'SExtractor++'


def generate_report(output, simulation, image, target, reference,
                    target_columns=None, reference_columns=None, target_flag_columns=None):
    """
    Generate a PDF comparing the target and the reference catalogs
    :param output:
        Path for the output PDF
    :param simulation:
        Original stuff simulation used (as returned by stuff.parse_stuff_list)
    :param image:
        Path for the rasterized image used for the detection
    :param target:
        Target catalog (Expects SExtractor++ output column names)
    :param target_columns:
        Columns to compare with the reference. Must be a list of tuples (value, error)
    :param reference_columns:
        Columns to compare with the target. Must be a list of tuples (value, error)
    :param target_flag_columns:
        Columns that contain flags for the target. Note that for the reference only FLAGS is used.
    :param reference:
        Reference catalog (Expectes SExtractor 2 column names)
    """
    if target_columns is None:
        target_columns = _default_target_columns
    if reference_columns is None:
        reference_columns = _default_reference_columns
    if target_flag_columns is None:
        target_flag_columns = _default_target_flag_columns

    stars, galaxies, kdtree = simulation
    expected_mags = np.append(stars.mag, galaxies.mag)
    target_closest = stuff.get_closest(target, kdtree)
    ref_closest = stuff.get_closest(reference, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY')

    img = fits.open(image)[0].data

    with PdfPages(output) as pdf:
        # Location with the image on the background
        fig = _plot_location(img, reference, target)
        pdf.savefig(fig)
        plt.close(fig)

        # Distances
        fig = _plot_distances(fig, ref_closest, target_closest)
        pdf.savefig(fig)
        plt.close(fig)

        # Column sets
        # Columns within a single set share the Y axis, so they are easier to compare
        for ref_set, target_set in zip(reference_columns, target_columns):
            figures = _plot_column_set(
                expected_mags, ref_set, target_set, ref_closest, target_closest, reference, target
            )

            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)

        # Flags
        figures = _plot_flags(img, reference, target, target_flag_columns)
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)


def _plot_location(img, reference, target):
    fig = plt.figure(figsize=_page_size)
    ax = fig.subplots()
    ax.set_title('Location')
    ax.imshow(img, cmap=_img_cmap, norm=_img_norm)
    ax.scatter(
        reference['X_IMAGE'], reference['Y_IMAGE'],
        marker='o', label=_ref_label, alpha=0.5
    )
    ax.scatter(
        target['pixel_centroid_x'], target['pixel_centroid_y'],
        marker='.', label=_target_label, alpha=0.5
    )
    ax.legend()
    return fig


def _plot_distances(fig, ref_closest, target_closest):
    fig = plt.figure(figsize=_page_size)
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title(f'Distances for {_target_label}')
    _, bins, _ = ax.hist(target_closest['dist'], bins=50)
    ax = fig.add_subplot(2, 1, 2)
    ax.set_title(f'Distances for {_ref_label}')
    ax.hist(ref_closest['dist'], bins=bins)
    return fig


def _plot_column_set(expected_mags, ref_set, target_set, ref_closest, target_closest, reference, target):
    figures = []
    shared_ax_y = None
    shared_ax_y_err = None
    shared_ax_y_diff = None
    for (ref_colname, ref_err_colname), (target_colname, target_err_colname) in zip(ref_set, target_set):
        is_magnitude = 'mag' in target_colname

        try:
            ref_val = get_column(reference, ref_colname)
            ref_err = get_column(reference, ref_err_colname)
            target_val = get_column(target, target_colname)
            target_err = get_column(target, target_err_colname)
        except ValueError:
            continue

        fig = plt.figure(figsize=_page_size)
        figures.append(fig)

        fig.subplots_adjust(left=0.07, right=0.93, hspace=0.0, wspace=0.2)

        gridspec = GridSpec(4, 1)
        ax1 = fig.add_subplot(gridspec.new_subplotspec((0, 0), 2), sharey=shared_ax_y)
        shared_ax_y = ax1
        ax1.set_title(f'{ref_colname} vs {target_colname}')

        ax1.scatter(
            expected_mags[ref_closest['source']], ref_val,
            marker='o', label=_ref_label
        )

        ax1.scatter(
            expected_mags[target_closest['source']], target_val,
            marker='.', label=_target_label
        )

        ax1.set_ylabel('Measured magnitude')
        ax1.grid(True, linestyle=':')
        ax1.set_xticklabels([])
        ax1.legend()

        if is_magnitude:
            ax1.spines['bottom'].set_linestyle('--')
            ax2 = fig.add_subplot(gridspec.new_subplotspec((2, 0), 1), sharey=shared_ax_y_diff)
            ax2.set_facecolor('whitesmoke')
            ax2.spines['top'].set_visible(False)
            shared_ax_y_diff = ax2

            ref_diff = ref_val - expected_mags[ref_closest['source']]
            target_diff = target_val - expected_mags[target_closest['source']]

            ax2.scatter(expected_mags[ref_closest['source']], ref_diff, marker='o')
            ax2.scatter(expected_mags[ref_closest['source']], target_diff, marker='.')
            ax2.set_ylabel('$\Delta$')
            ax2.set_xticklabels([])
            ax2.axhline(0, color='gray')

            ax2.grid(True, linestyle=':')

            ax3 = fig.add_subplot(gridspec.new_subplotspec((3, 0), 1), sharey=shared_ax_y_err)
        else:
            ax3 = fig.add_subplot(gridspec.new_subplotspec((2, 0), 2), sharey=shared_ax_y_err)
        ax3.set_facecolor('oldlace')
        shared_ax_y_err = ax3
        ax3.scatter(
            expected_mags[ref_closest['source']], ref_err,
            marker='o'
        )
        ax3.scatter(
            expected_mags[target_closest['source']], target_err,
            marker='.'
        )
        ax3.set_ylabel('Catalog error')
        ax3.set_xlabel('Real magnitude')
        ax3.grid(True, linestyle=':')

        figures.append(fig)
    return figures


def _plot_flags(img, reference, target, target_flag_columns):
    figures = []
    for flag_col in target_flag_columns:
        try:
            target_flags = get_column(target, flag_col)
        except ValueError:
            continue

        fig = plt.figure(figsize=_page_size)
        figures.append(fig)
        ax = fig.add_subplot(1, 2, 1)
        markers = cycle(['1', '2', '3', '4'])

        ax.set_title(f'{_ref_label} FLAGS')
        ax.imshow(img, cmap=_img_cmap, norm=_img_norm)
        for flag in stuff.Sex2SourceFlags:
            flag_filter = (reference['FLAGS'] & int(flag)).astype(np.bool)
            if flag_filter.any():
                ax.scatter(
                    reference[flag_filter]['X_IMAGE'], reference[flag_filter]['Y_IMAGE'],
                    label=flag, marker=next(markers)
                )
        ax.legend()

        ax = fig.add_subplot(1, 2, 2)
        markers = cycle(['1', '2', '3', '4'])
        ax.set_title(f'{_target_label} {flag_col}')
        ax.imshow(img, cmap=_img_cmap, norm=_img_norm)
        for flag in stuff.SourceFlags:
            flag_filter = (target_flags & int(flag)).astype(np.bool)
            if flag_filter.any():
                ax.scatter(
                    target[flag_filter]['pixel_centroid_x'], target[flag_filter]['pixel_centroid_y'],
                    label=flag, marker=next(markers)
                )
        ax.legend()
    return figures
