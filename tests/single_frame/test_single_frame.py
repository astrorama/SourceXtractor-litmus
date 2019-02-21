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

from util import stuff, get_column, plot


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
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels',
            detection_image=datafiles / 'sim09' / 'img' / 'sim09_r_01.fits',
            python_config_file=datafiles / 'sim09' / 'sim09_single.py'
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= signal_to_noise_ratio
    return np.sort(catalog[bright_filter], order=('world_centroid_alpha', 'world_centroid_delta'))


def test_detection(single_frame_catalog, reference):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(single_frame_catalog) > 0
    assert len(single_frame_catalog) == len(reference)


def test_location(single_frame_catalog, reference, stuff_simulation, tolerances):
    """
    The detections should be at least as close as the ref to the truth.
    Single frame simulations are in pixel coordinates.
    """
    _, _, kdtree = stuff_simulation

    det_closest = stuff.get_closest(
        single_frame_catalog, kdtree
    )
    ref_closest = stuff.get_closest(
        reference, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY'
    )

    assert np.mean(det_closest['dist']) <= np.mean(ref_closest['dist']) * (1 + tolerances['distance'])


@pytest.mark.parametrize(
    ['flux_column', 'reference_flux_column'], [
        [['isophotal_flux', 'isophotal_flux_err'], ['FLUX_ISO', 'FLUXERR_ISO']],
        [['auto_flux', 'auto_flux_err'], ['FLUX_AUTO', 'FLUXERR_AUTO']],
        [['aperture_flux:0:0', 'aperture_flux_err:0:0'], ['FLUX_APER:0', 'FLUXERR_APER:0']],
        [['aperture_flux:0:1', 'aperture_flux_err:0:1'], ['FLUX_APER:1', 'FLUXERR_APER:1']],
        [['aperture_flux:0:2', 'aperture_flux_err:0:2'], ['FLUX_APER:2', 'FLUXERR_APER:2']],
    ]
)
def test_fluxes(single_frame_catalog, reference, flux_column, reference_flux_column, tolerances):
    """
    Cross-validate flux columns. The measured fluxes and errors should be close to the ref.
    """
    target_flux = get_column(single_frame_catalog, flux_column[0])
    reference_flux = get_column(reference, reference_flux_column[0])
    target_flux_err = get_column(single_frame_catalog, flux_column[1])
    reference_flux_err = get_column(reference, reference_flux_column[1])

    relative_flux_diff = np.abs((target_flux - reference_flux) / target_flux)
    relative_flux_err_diff = np.abs((target_flux_err - reference_flux_err) / target_flux_err)
    assert np.nanmedian(relative_flux_diff) < tolerances['flux']
    assert np.nanmedian(relative_flux_err_diff) < tolerances['flux']


@pytest.mark.parametrize(
    ['mag_column', 'reference_mag_column'], [
        [['isophotal_mag', 'isophotal_mag_err'], ['MAG_ISO', 'MAGERR_ISO']],
        [['auto_mag', 'auto_mag_err'], ['MAG_AUTO', 'MAGERR_AUTO']],
        [['aperture_mag:0:0', 'aperture_mag_err:0:0'], ['MAG_APER:0', 'MAGERR_APER:0']],
        [['aperture_mag:0:1', 'aperture_mag_err:0:1'], ['MAG_APER:1', 'MAGERR_APER:1']],
        [['aperture_mag:0:2', 'aperture_mag_err:0:2'], ['MAG_APER:2', 'MAGERR_APER:2']],
    ]
)
def test_magnitude(single_frame_catalog, reference, mag_column, reference_mag_column, stuff_simulation, tolerances):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    """
    stars, galaxies, kdtree = stuff_simulation
    expected_mags = np.append(stars.mag, galaxies.mag)

    det_closest = stuff.get_closest(
        single_frame_catalog, kdtree
    )
    ref_closest = stuff.get_closest(
        reference, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY'
    )

    det_mag = get_column(single_frame_catalog[det_closest['catalog']], mag_column[0])
    ref_mag = get_column(reference[ref_closest['catalog']], reference_mag_column[0])
    relative_mag_diff = np.abs((expected_mags[det_closest['source']] - det_mag) / det_mag)
    relative_ref_diff = np.abs((expected_mags[ref_closest['source']] - ref_mag) / ref_mag)

    assert np.median(relative_mag_diff) <= np.median(relative_ref_diff) * (1 + tolerances['magnitude'])


def test_generate_report(single_frame_catalog, reference, stuff_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    plot.generate_report(
        module_output_area / 'report.pdf', stuff_simulation, datafiles / 'sim09' / 'img' / 'sim09_r_01.fits',
        single_frame_catalog, reference
    )
