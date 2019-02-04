import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

from . import stuff, get_column

_default_target_columns = [
    ('isophotal_mag', 'isophotal_mag_err'),
    ('auto_mag', 'auto_mag_err'),
    ('aperture_mag:0:0', 'aperture_mag_err:0:0'),
    ('aperture_mag:0:1', 'aperture_mag_err:0:1'),
    ('aperture_mag:0:2', 'aperture_mag_err:0:2')
]
_default_reference_columns = [
    ('MAG_ISO', 'MAGERR_ISO'),
    ('MAG_AUTO', 'MAGERR_AUTO'),
    ('MAG_APER:0', 'MAGERR_APER:0'),
    ('MAG_APER:1', 'MAGERR_APER:1'),
    ('MAG_APER:2', 'MAGERR_APER:2')
]
_default_target_flag_columns = [
    'source_flags', 'auto_flags', 'aperture_flags:0:0', 'aperture_flags:0:1', 'aperture_flags:0:2'
]
_page_size = (11.7, 8.3)
_img_cmap = plt.get_cmap('Greys_r')
_img_norm = colors.SymLogNorm(10)


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
        plt.figure(figsize=_page_size)
        plt.title('Location')
        plt.imshow(img, cmap=_img_cmap, norm=_img_norm)
        plt.scatter(
            reference['X_IMAGE'], reference['Y_IMAGE'],
            marker='o', label='Reference', alpha=0.5
        )
        plt.scatter(
            target['pixel_centroid_x'], target['pixel_centroid_y'],
            marker='.', label='Output', alpha=0.5
        )
        plt.legend()
        pdf.savefig()

        # Magnitudes
        for (ref_col, ref_err_col), (target_col, target_err_col) in zip(reference_columns, target_columns):
            ref_mag = get_column(reference, ref_col)
            ref_mag_err = get_column(reference, ref_err_col)
            det_mag = get_column(target, target_col)
            det_mag_err = get_column(target, target_err_col)

            plt.figure(figsize=_page_size)
            plt.subplots_adjust(left=0.07, right=0.93, hspace=0.0, wspace=0.2)

            ax1 = plt.subplot2grid((4, 1), (0, 0), 2)
            ax1.set_title(f'{ref_col} vs {target_col}')

            ax1.scatter(
                expected_mags[target_closest['source']], ref_mag,
                marker='o', label='Reference'
            )

            ax1.scatter(
                expected_mags[ref_closest['source']], det_mag,
                marker='.', label='Output'
            )

            ax1.set_ylabel('Measured magnitude')
            ax1.grid(True, linestyle=':')
            ax1.set_xticklabels([])
            ax1.legend()

            ax2 = plt.subplot2grid((4, 1), (2, 0), 1)
            ax2.scatter(
                expected_mags[target_closest['source']], expected_mags[target_closest['source']] - ref_mag,
                marker='o'
            )
            ax2.scatter(
                expected_mags[ref_closest['source']], expected_mags[ref_closest['source']] - det_mag,
                marker='.'
            )
            ax2.set_ylabel('$\Delta$')
            ax2.set_xticklabels([])
            ax2.grid(True, linestyle=':')

            ax3 = plt.subplot2grid((4, 1), (3, 0), 1)
            ax3.scatter(
                expected_mags[ref_closest['source']], ref_mag_err,
                marker='o'
            )
            ax3.scatter(
                expected_mags[ref_closest['source']], det_mag_err,
                marker='.'
            )
            ax3.set_ylabel('Measured error')
            ax3.set_xlabel('Real magnitude')
            ax3.grid(True, linestyle=':')

            pdf.savefig()

        # Flags
        for flag_col in target_flag_columns:
            target_flags = get_column(target, flag_col)
            plt.figure(figsize=_page_size)
            plt.subplot(1, 2, 1)

            plt.title('Flags for the reference')
            plt.imshow(img, cmap=_img_cmap, norm=_img_norm)
            for flag in stuff.SourceFlags:
                flag_filter = (reference['FLAGS'] & int(flag)).astype(np.bool)
                if flag_filter.any():
                    plt.scatter(
                        reference[flag_filter]['X_IMAGE'], reference[flag_filter]['Y_IMAGE'],
                        label=flag, alpha=0.5
                    )
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.title(f'Output {flag_col}')
            plt.imshow(img, cmap=_img_cmap, norm=_img_norm)
            for flag in stuff.SourceFlags:
                flag_filter = (target_flags & int(flag)).astype(np.bool)
                if flag_filter.any():
                    plt.scatter(
                        target[flag_filter]['pixel_centroid_x'], target[flag_filter]['pixel_centroid_y'],
                        label=flag, alpha=0.5
                    )
            plt.legend()

            pdf.savefig()
