#
# This test is almost identical to test_single_frame but it does *NOT*
# use the measurement configuration file, and limits itself to measure
# the AutoPhotometry and IsophotalFlux on the detection image.
#
# The reason for this two separate runs is that the pre-release emitted different
# when the measurement frame matched the detection image, and when it was configured
# separately
#
import os

import numpy as np
import pytest

from util import stuff, plot
from astropy.table import Table


@pytest.fixture
def single_frame_catalog(sextractorxx, datafiles, module_output_area, signal_to_noise_ratio):
    """
    Run sextractorxx on a single frame. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise, and sorted by location, so cross-matching is easier
    """
    sextractorxx.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags',
            detection_image=datafiles / 'sim09' / 'sim09_r_01.fits',
            psf_file=datafiles / 'sim09' / 'sim09_r_01.psf',
            python_config_file=None
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= signal_to_noise_ratio
    return np.sort(catalog[bright_filter], order=('world_centroid_alpha', 'world_centroid_delta'))


def test_detection(single_frame_catalog, reference):
    """
    Test that the number of results matches the reference, and that they are reasonably close
    """
    assert len(single_frame_catalog) > 0
    assert len(single_frame_catalog) == len(reference)


def test_location(single_frame_catalog, reference, stuff_simulation):
    """
    The detections should be at least as close as the reference to the truth.
    Single frame simulations are in pixel coordinates.
    """
    _, _, kdtree = stuff_simulation

    det_closest = stuff.get_closest(
        single_frame_catalog, kdtree
    )
    ref_closest = stuff.get_closest(
        reference, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY'
    )

    assert np.mean(det_closest['dist']) <= np.mean(ref_closest['dist'])


@pytest.mark.parametrize(
    ['mag_column', 'reference_mag_column'], [
        [['isophotal_mag', 'isophotal_mag_err'], ['MAG_ISO', 'MAGERR_ISO']],
        [['auto_mag', 'auto_mag_err'], ['MAG_AUTO', 'MAGERR_AUTO']],
    ]
)
def test_magnitude(single_frame_catalog, reference, mag_column, reference_mag_column, stuff_simulation, tolerances):
    """
    Cross-validate flux columns. The measured magnitudes should be at least as close
    to the truth as the reference catalog (within a tolerance).
    """
    stars, galaxies, kdtree = stuff_simulation
    expected_mags = np.append(stars.mag, galaxies.mag)

    det_closest = stuff.get_closest(
        single_frame_catalog, kdtree
    )
    ref_closest = stuff.get_closest(
        reference, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY'
    )

    det_mag = single_frame_catalog[det_closest['catalog']][mag_column[0]]
    ref_mag = reference[ref_closest['catalog']][reference_mag_column[0]]
    det_mag_diff = expected_mags[det_closest['source']] - det_mag
    ref_mag_diff = expected_mags[ref_closest['source']] - ref_mag
    assert np.mean(np.abs(det_mag_diff)) <= np.mean(np.abs(ref_mag_diff)) * tolerances['magnitude']


def test_generate_report(single_frame_catalog, reference, stuff_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    plot.generate_report(
        module_output_area / 'report.pdf', stuff_simulation, datafiles / 'sim09' / 'sim09_r_01.fits',
        single_frame_catalog, reference,
        target_columns=[('isophotal_mag', 'isophotal_mag_err'), ('auto_mag', 'auto_mag_err')],
        reference_columns=[('MAG_ISO', 'MAGERR_ISO'), ('MAG_AUTO', 'MAGERR_AUTO')],
        target_flag_columns=['source_flags', 'auto_flags']
    )
