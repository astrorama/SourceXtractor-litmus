import os

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.catalog import get_column
from util.image import Image
from util.validation import CrossValidation


@pytest.fixture
def coadded_catalog(sextractorxx, datafiles, module_output_area, tolerances):
    """
    Run sextractorxx on a coadded single frame. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise.
    """
    sextractorxx.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels',
            detection_image=datafiles / 'sim09' / 'img' / 'sim09_r.fits',
            weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits',
            weight_type='weight',
            weight_absolute=True,
            psf_filename=datafiles / 'sim09' / 'psf' / 'sim09_r.psf'
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    for nan_col in ['isophotal_mag', 'auto_mag', 'isophotal_mag_err', 'auto_mag_err']:
        catalog[nan_col][catalog[nan_col] >= 99.] = np.nan
    return catalog[bright_filter]


@pytest.fixture
def coadded_frame_cross(coadded_catalog, sim09_r_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim09' / 'img' / 'sim09_r.fits',
        weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits'
    )
    cross = CrossValidation(image, sim09_r_simulation, max_dist=tolerances['distance'])
    return cross(coadded_catalog['pixel_centroid_x'], coadded_catalog['pixel_centroid_y'])


def test_detection(coadded_frame_cross, sim09_r_cross):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(coadded_frame_cross.stars_found) >= len(sim09_r_cross.stars_found)
    assert len(coadded_frame_cross.galaxies_found) >= len(sim09_r_cross.galaxies_found)


@pytest.mark.parametrize(
    ['mag_column', 'reference_mag_column'], [
        [['isophotal_mag', 'isophotal_mag_err'], ['MAG_ISO', 'MAGERR_ISO']],
        [['auto_mag', 'auto_mag_err'], ['MAG_AUTO', 'MAGERR_AUTO']]
    ]
)
def test_magnitude(coadded_catalog, sim09_r_reference, mag_column, reference_mag_column, coadded_frame_cross,
                   sim09_r_cross, tolerances):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    We use only the hits, and ignore the detections that are a miss.
    """

    # We use only those sources that are a hit, and do not bother to compare
    # others
    catalog_hits = coadded_catalog[coadded_frame_cross.all_catalog]
    ref_hits = sim09_r_reference[sim09_r_cross.all_catalog]

    catalog_mag = get_column(catalog_hits, mag_column[0])
    ref_mag = get_column(ref_hits, reference_mag_column[0])

    catalog_mag_diff = catalog_mag - coadded_frame_cross.all_magnitudes
    ref_mag_diff = ref_mag - sim09_r_cross.all_magnitudes

    assert np.median(catalog_mag_diff) <= np.median(ref_mag_diff) * (1 + tolerances['magnitude'])


def test_generate_report(coadded_catalog, sim09_r_reference, sim09_r_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    plot.generate_report(
        module_output_area / 'report.pdf', sim09_r_simulation,
        datafiles / 'sim09' / 'img' / 'sim09_r.fits',
        coadded_catalog, sim09_r_reference,
        weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits',
    )
