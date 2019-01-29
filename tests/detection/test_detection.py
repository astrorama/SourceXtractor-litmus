from astropy.table import Table

import pytest


@pytest.mark.parametrize(
    'dataset', ['sim09'],
)
def test_detection(sextractorxx, datafiles, dataset):
    run = sextractorxx(
        output_properties='WorldCentroid',
        detection_image=datafiles / dataset / 'detection.fits'
    )
    assert run.exit_code == 0

    table = Table.read(sextractorxx.get_output_catalog())
    assert len(table['world_centroid_alpha']) > 0
