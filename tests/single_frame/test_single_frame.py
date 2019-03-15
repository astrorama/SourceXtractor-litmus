#
# This test is almost identical to test_single_frame_no_py but it *DOES*
# use the measurement configuration file,.
#
# The reason for this two separate runs is that the pre-release emitted different
# when the measurement frame matched the detection image, and when it was configured
# separately
#
import os

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.catalog import get_column
from util.validation import CrossValidation


@pytest.fixture
def single_frame_catalog(sextractorxx, datafiles, module_output_area, tolerances):
    """
    Run sextractorxx on a single frame. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise.
    """
    sextractorxx.set_output_directory(module_output_area)

    detection_image = datafiles / 'sim09' / 'img' / 'sim09_r_01.fits'

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels,AperturePhotometry',
            detection_image=detection_image,
            python_config_file=datafiles / 'sim09' / 'sim09_single.py'
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture
def single_frame_cross(single_frame_catalog, sim09_r_simulation, datafiles, tolerances):
    detection_image = datafiles / 'sim09' / 'img' / 'sim09_r_01.fits'
    cross = CrossValidation(detection_image, sim09_r_simulation, max_dist=tolerances['distance'])
    return cross(single_frame_catalog['pixel_centroid_x'], single_frame_catalog['pixel_centroid_y'])


def test_detection(single_frame_cross, sim09_r_01_cross):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(single_frame_cross.stars_found) >= len(sim09_r_01_cross.stars_found)
    assert len(single_frame_cross.galaxies_found) >= len(sim09_r_01_cross.galaxies_found)


@pytest.mark.parametrize(
    ['mag_column', 'reference_mag_column'], [
        [['isophotal_mag', 'isophotal_mag_err'], ['MAG_ISO', 'MAGERR_ISO']],
        [['auto_mag', 'auto_mag_err'], ['MAG_AUTO', 'MAGERR_AUTO']],
        [['aperture_mag:0:0', 'aperture_mag_err:0:0'], ['MAG_APER:0', 'MAGERR_APER:0']],
        [['aperture_mag:0:1', 'aperture_mag_err:0:1'], ['MAG_APER:1', 'MAGERR_APER:1']],
        [['aperture_mag:0:2', 'aperture_mag_err:0:2'], ['MAG_APER:2', 'MAGERR_APER:2']],
    ]
)
def test_magnitude(single_frame_catalog, single_frame_cross, sim09_r_01_reference, sim09_r_01_cross,
                   mag_column, reference_mag_column, tolerances):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    We use only the hits, and ignore the detections that are a miss.
    """

    # We use only those sources that are a hit, and do not bother to compare
    # others
    catalog_hits = single_frame_catalog[single_frame_cross.all_catalog]
    ref_hits = sim09_r_01_reference[sim09_r_01_cross.all_catalog]

    catalog_mag = get_column(catalog_hits, mag_column[0])
    ref_mag = get_column(ref_hits, reference_mag_column[0])

    catalog_mag_diff = catalog_mag - single_frame_cross.all_magnitudes
    ref_mag_diff = ref_mag - sim09_r_01_cross.all_magnitudes

    assert np.median(catalog_mag_diff) <= np.median(ref_mag_diff) * (1 + tolerances['magnitude'])


def test_generate_report(single_frame_catalog, sim09_r_01_reference, sim09_r_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    plot.generate_report(
        module_output_area / 'report.pdf', sim09_r_simulation, datafiles / 'sim09' / 'img' / 'sim09_r_01.fits',
        single_frame_catalog, sim09_r_01_reference
    )
