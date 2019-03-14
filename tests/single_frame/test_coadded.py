import os

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.catalog import get_column


@pytest.fixture
def coadded_catalog(sextractorxx, datafiles, module_output_area, signal_to_noise_ratio):
    """
    Run sextractorxx on a coadded single frame. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise, and sorted by location, so cross-matching is easier.
    Note that we pass whe weight also here, as it is used by the isophotal plugin.
    For the others (auto, aperture), it is used the weight configured on the measurement frame
    (so, in the Python config file).
    """
    sextractorxx.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels,AperturePhotometry',
            detection_image=datafiles / 'sim09' / 'img' / 'sim09_r.fits',
            weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits',
            python_config_file=datafiles / 'sim09' / 'sim09_single_weights.py',
            weight_type='weight',
            weight_absolute=True
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= signal_to_noise_ratio
    for nan_col in ['isophotal_mag', 'auto_mag', 'isophotal_mag_err', 'auto_mag_err']:
        catalog[nan_col][catalog[nan_col] >= 99.] = np.nan
    return np.sort(catalog[bright_filter], order=('world_centroid_alpha', 'world_centroid_delta'))


def test_detection(coadded_catalog, coadded_reference):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(coadded_catalog) > 0
    assert len(coadded_catalog) == len(coadded_reference)


def test_location(coadded_catalog, coadded_reference, stuff_simulation, tolerances):
    """
    The detections should be at least as close as the ref to the truth.
    Single frame simulations are in pixel coordinates.
    """
    det_closest = stuff_simulation.get_closest(
        coadded_catalog['world_centroid_alpha'], coadded_catalog['world_centroid_delta']
    )
    ref_closest = stuff_simulation.get_closest(
        coadded_reference['ALPHA_SKY'], coadded_reference['DELTA_SKY']
    )

    assert np.mean(det_closest.distance) <= np.mean(ref_closest.distance) * (1 + tolerances['distance'])


@pytest.mark.parametrize(
    ['flux_column', 'reference_flux_column'], [
        [['isophotal_flux', 'isophotal_flux_err'], ['FLUX_ISO', 'FLUXERR_ISO']],
        [['auto_flux', 'auto_flux_err'], ['FLUX_AUTO', 'FLUXERR_AUTO']]
    ]
)
def test_fluxes(coadded_catalog, coadded_reference, flux_column, reference_flux_column, tolerances):
    """
    Cross-validate flux columns. The measured fluxes and errors should be close to the ref.
    """
    target_flux = get_column(coadded_catalog, flux_column[0])
    reference_flux = get_column(coadded_reference, reference_flux_column[0])
    target_flux_err = get_column(coadded_catalog, flux_column[1])
    reference_flux_err = get_column(coadded_reference, reference_flux_column[1])

    relative_flux_diff = np.abs((target_flux - reference_flux) / target_flux)
    relative_flux_err_diff = np.abs((target_flux_err - reference_flux_err) / target_flux_err)
    assert np.nanmedian(relative_flux_diff) < tolerances['flux']
    assert np.nanmedian(relative_flux_err_diff) < tolerances['flux']


@pytest.mark.parametrize(
    ['mag_column', 'reference_mag_column'], [
        [['isophotal_mag', 'isophotal_mag_err'], ['MAG_ISO', 'MAGERR_ISO']],
        [['auto_mag', 'auto_mag_err'], ['MAG_AUTO', 'MAGERR_AUTO']]
    ]
)
def test_magnitude(coadded_catalog, coadded_reference, mag_column, reference_mag_column, stuff_simulation, tolerances):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    """
    det_closest = stuff_simulation.get_closest(
        coadded_catalog['world_centroid_alpha'], coadded_catalog['world_centroid_delta']
    )
    ref_closest = stuff_simulation.get_closest(
        coadded_reference['ALPHA_SKY'], coadded_reference['DELTA_SKY']
    )

    det_mag = get_column(coadded_catalog[det_closest.catalog], mag_column[0])
    ref_mag = get_column(coadded_reference[ref_closest.catalog], reference_mag_column[0])
    relative_mag_diff = np.abs((det_closest.magnitude - det_mag) / det_mag)
    relative_ref_diff = np.abs((ref_closest.magnitude - ref_mag) / ref_mag)

    assert np.median(relative_mag_diff) <= np.median(relative_ref_diff) * (1 + tolerances['magnitude'])


def test_generate_report(coadded_catalog, coadded_reference, stuff_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    plot.generate_report(
        module_output_area / 'report.pdf', stuff_simulation,
        datafiles / 'sim09' / 'img' / 'sim09_r.fits',
        coadded_catalog, coadded_reference,
        weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits'
    )
