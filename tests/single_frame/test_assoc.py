from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from util.matching import CrossMatching, intersect


@pytest.fixture(scope='module')
def assoc_run(sourcextractor, datafiles, module_output_area, tolerances):
    """
    Run sourcextractor on a single frame ussing assoc mode
    """
    sourcextractor.set_output_directory(module_output_area)

    properties = ['SourceIDs',
                  'PixelCentroid',
                  'WorldCentroid',
                  'AssocMode']

    run = sourcextractor(
        output_properties=','.join(properties),
        detection_image=datafiles / 'sim12' / 'img' / 'sim12_r.fits.gz',
        psf_filename=datafiles / 'sim12' / 'psf' / 'sim12_r.psf',
        assoc_catalog=datafiles / 'sim12' / 'sim12_assoc.fits',
        assoc_mode='NEAREST',
        assoc_filter='MATCHED',
        assoc_coord_type='WORLD',
        assoc_copy='4,5'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    return SimpleNamespace(run=run, catalog=catalog)


@pytest.fixture(scope='module')
def assoc_catalog(assoc_run):
    return assoc_run.catalog


@pytest.fixture(scope='module')
def assoc_cross(assoc_catalog, sim12_r_simulation, datafiles, tolerances):
    detection_image = datafiles / 'sim12' / 'img' / 'sim12_r.fits.gz'
    cross = CrossMatching(detection_image, sim12_r_simulation, max_dist=tolerances['distance'])
    return cross(assoc_catalog['pixel_centroid_x'], assoc_catalog['pixel_centroid_y'])


def test_detection(assoc_catalog, assoc_cross, sim12_r_reference, sim12_r_cross):
    """
    Test that the number of results and the assoc copied values match expectations
    """
    assert len(assoc_catalog) <= 73
    assert np.all(assoc_catalog['assoc_match'])
    assert assoc_catalog['assoc_values'].shape[1] == 2

    assoc_intersect, ref_intersect = intersect(assoc_cross, sim12_r_cross)
    catalog_hits = assoc_catalog[assoc_cross.all_catalog[assoc_intersect]]
    ref_hits = sim12_r_reference[sim12_r_cross.all_catalog[ref_intersect]]

    # SNR_WIN
    np.testing.assert_allclose(catalog_hits['assoc_values'][:, 0], ref_hits['SNR_WIN'])
    # ELONGATION
    np.testing.assert_allclose(catalog_hits['assoc_values'][:, 1], ref_hits['ELONGATION'])
