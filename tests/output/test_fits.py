import os
import numpy as np
from astropy.io import fits


def test_fits_output_file(sextractorxx, datafiles):
    """
    Run SExtractor asking for FITS output on a file
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    assert os.path.exists(single_source_fits)

    output_catalog = sextractorxx.get_output_directory() / 'output.fits'

    run = sextractorxx(
        detection_image=single_source_fits,
        output_file_format='FITS',
        output_file=output_catalog,
        output_properties='SourceIDs,PixelCentroid'
    )
    assert run.exit_code == 0
    assert os.path.exists(output_catalog)

    hdul = fits.open(output_catalog)
    # Primary + Table
    assert len(hdul) == 2

    table = hdul[1].data
    # Source ID
    assert table['source_id'][0] == 1
    # Coordinates
    assert np.isclose(table['pixel_centroid_x'][0], 21.5819)
    assert np.isclose(table['pixel_centroid_y'][0], 24.0353)
