import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

from . import stuff, get_column

_default_target_aper_columns = ['isophotal_mag', 'auto_mag', 'aperture_mag:0:0', 'aperture_mag:0:1', 'aperture_mag:0:2']
_default_reference_aper_columns = ['MAG_ISO', 'MAG_AUTO', 'MAG_APER:0', 'MAG_APER:1', 'MAG_APER:2']


def generate_report(output, simulation, image, target, reference,
                    target_aper_columns=None, reference_aper_columns=None):
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
    :param reference:
        Reference catalog (Expectes SExtractor 2 column names)
    """
    if target_aper_columns is None:
        target_aper_columns = _default_target_aper_columns
    if reference_aper_columns is None:
        reference_aper_columns = _default_reference_aper_columns

    stars, galaxies, kdtree = simulation
    expected_mags = np.append(stars.mag, galaxies.mag)
    target_closest = stuff.get_closest(target, kdtree)
    ref_closest = stuff.get_closest(reference, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY')

    img = fits.open(image)[0].data

    with PdfPages(output) as pdf:
        # Location with the image on the background
        plt.figure(figsize=(11.7, 8.3))
        plt.title('Location')
        plt.imshow(img, cmap=plt.get_cmap('Greys_r'), norm=colors.SymLogNorm(100))
        plt.scatter(
            reference['X_IMAGE'], reference['Y_IMAGE'],
            marker='^', label='Reference', alpha=0.5
        )
        plt.scatter(
            target['pixel_centroid_x'], target['pixel_centroid_y'],
            marker='v', label='Output', alpha=0.5
        )
        plt.legend()
        pdf.savefig()

        # Magnitudes
        for ref_col, det_col in zip(reference_aper_columns, target_aper_columns):
            ref_mag = get_column(reference, ref_col)
            det_mag = get_column(target, det_col)

            plt.figure(figsize=(11.7, 8.3))
            plt.subplots_adjust(left=0.07, right=0.93, hspace=0.0, wspace=0.2)

            ax1 = plt.subplot2grid((3, 1), (0, 0), 2)
            ax1.set_title(f'{ref_col} vs {det_col}')

            ax1.scatter(
                expected_mags[target_closest['source']], ref_mag,
                marker='^', label='Reference'
            )

            ax1.scatter(
                expected_mags[ref_closest['source']], det_mag,
                marker='v', label='Output'
            )

            ax1.set_ylabel('Measured magnitude')
            ax1.legend()

            ax2 = plt.subplot2grid((3, 1), (2, 0), 1)
            ax2.scatter(
                expected_mags[target_closest['source']], expected_mags[target_closest['source']] - ref_mag,
                marker='^', label='Reference'
            )
            ax2.scatter(
                expected_mags[ref_closest['source']], expected_mags[ref_closest['source']] - det_mag,
                marker='v', label='Output'
            )
            ax2.set_ylabel('$\Delta$')
            ax2.set_xlabel('Real magnitude')

            pdf.savefig()
