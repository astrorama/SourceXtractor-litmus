from pathlib import Path
from astropy.table import Table

import pytest


@pytest.mark.parametrize(
    'dataset', ['sim09'],
)
def test_detection(sextractorxx, datafiles, dataset):
    run = sextractorxx.run_with_config(
        output_properties='PixelCentroid',
        detection_image=datafiles / dataset / 'detection.fits',
        psf_file=datafiles / dataset / 'detection.psf'
    )
    assert run.exit_code == 0

    table = Table.read(sextractorxx.get_output_catalog())
    assert len(table['pixel_centroid_x']) > 0
