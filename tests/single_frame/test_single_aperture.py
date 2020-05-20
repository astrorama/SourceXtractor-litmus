import numpy as np
import pytest
from astropy.table import Table

from util.catalog import get_column
from util.matching import CrossMatching, intersect


@pytest.fixture(scope='module')
def single_frame_catalog(sourcextractor, datafiles, module_output_area, tolerances):
    """
    Run sourcextractor on a single frame. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise.
    """
    sourcextractor.set_output_directory(module_output_area)

    detection_image = datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz'

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels,AperturePhotometry',
        detection_image=detection_image,
        python_config_file=datafiles / 'sim11' / 'sim11_single_aperture.py'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='module')
def single_frame_cross(single_frame_catalog, sim11_r_simulation, datafiles, tolerances):
    detection_image = datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz'
    cross = CrossMatching(detection_image, sim11_r_simulation, max_dist=tolerances['distance'])
    return cross(single_frame_catalog['pixel_centroid_x'], single_frame_catalog['pixel_centroid_y'])


@pytest.mark.parametrize(
    ['flux_column', 'reference_flux_column'], [
        [['isophotal_flux', 'isophotal_flux_err'], ['FLUX_ISO', 'FLUXERR_ISO']],
        [['aperture_flux', 'aperture_flux_err'], ['FLUX_APER:0', 'FLUXERR_APER:0']],
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

    assert catalog_flux.shape == (len(catalog_flux),)

    catalog_dist = np.sqrt((catalog_flux - real_flux) ** 2 / catalog_flux_err ** 2)
    ref_dist = np.sqrt((ref_flux - real_flux) ** 2 / ref_flux_err ** 2)

    assert np.median(catalog_dist - ref_dist) <= 0.
