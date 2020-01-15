#
# This test is almost identical to test_single_frame but it does *NOT*
# use the measurement configuration file, and limits itself to measure
# the AutoPhotometry and IsophotalFlux on the detection image.
#
# The reason for this two separate runs is that the pre-release emitted different
# when the measurement frame matched the detection image, and when it was configured
# separately
#
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.catalog import get_column
from util.validation import CrossValidation, intersect


@pytest.fixture(scope='module')
def single_frame_run(sourcextractor, datafiles, module_output_area, tolerances):
    """
    Run sourcextractor on a single frame. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise.
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels',
        detection_image=datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz',
        psf_filename=datafiles / 'sim11' / 'psf' / 'sim11_r_01.psf',
        python_config_file=None
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    return SimpleNamespace(run=run, catalog=catalog[bright_filter])


@pytest.fixture(scope='module')
def single_frame_catalog(single_frame_run):
    return single_frame_run.catalog


@pytest.fixture(scope='module')
def single_frame_cross(single_frame_catalog, sim11_r_simulation, datafiles, tolerances):
    detection_image = datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz'
    cross = CrossValidation(detection_image, sim11_r_simulation, max_dist=tolerances['distance'])
    return cross(single_frame_catalog['pixel_centroid_x'], single_frame_catalog['pixel_centroid_y'])


def test_detection(single_frame_cross, sim11_r_01_cross):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(single_frame_cross.stars_found) >= len(sim11_r_01_cross.stars_found)
    assert len(single_frame_cross.galaxies_found) >= len(sim11_r_01_cross.galaxies_found)


@pytest.mark.parametrize(
    ['flux_column', 'reference_flux_column'], [
        [['isophotal_flux', 'isophotal_flux_err'], ['FLUX_ISO', 'FLUXERR_ISO']],
        [['auto_flux', 'auto_flux_err'], ['FLUX_AUTO', 'FLUXERR_AUTO']],
    ]
)
def test_flux(single_frame_catalog, single_frame_cross, sim11_r_01_reference, sim11_r_01_cross,
              flux_column, reference_flux_column):
    """
    Cross-validate the flux columns.
    We use only the hits, and ignore the detections that are a miss.
    """
    catalog_intersect, ref_intersect = intersect(single_frame_cross, sim11_r_01_cross)
    catalog_hits = single_frame_catalog[single_frame_cross.all_catalog[catalog_intersect]]
    ref_hits = sim11_r_01_reference[sim11_r_01_cross.all_catalog[ref_intersect]]

    assert len(catalog_hits) == len(ref_hits)

    catalog_flux = get_column(catalog_hits, flux_column[0])
    catalog_flux_err = get_column(catalog_hits, flux_column[1])
    ref_flux = get_column(ref_hits, reference_flux_column[0])
    ref_flux_err = get_column(ref_hits, reference_flux_column[1])
    real_flux = sim11_r_01_cross.all_fluxes[ref_intersect]

    catalog_dist = np.sqrt((catalog_flux - real_flux) ** 2 / catalog_flux_err ** 2)
    ref_dist = np.sqrt((ref_flux - real_flux) ** 2 / ref_flux_err ** 2)

    assert np.median(catalog_dist - ref_dist) <= 0.


@pytest.mark.report
def test_generate_report(single_frame_run, sim11_r_01_reference, sim11_r_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    plot.generate_report(
        module_output_area / 'report.pdf', sim11_r_simulation, datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz',
        single_frame_run.catalog, sim11_r_01_reference, run=single_frame_run.run
    )
