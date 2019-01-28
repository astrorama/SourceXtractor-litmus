import os
import numpy as np
from astropy.io import fits


def test_fits_output_file(sextractorxx, datafiles, output_directory):
    """
    Run SExtractor asking for FITS output on a file
    """
    single_source_fits = datafiles / 'single_source.fits'
    assert os.path.exists(single_source_fits)

    output_catalog = output_directory / 'output.fits'

    run = sextractorxx.run(
        '--detection-image', single_source_fits,
        '--psf-fwhm', '2', '--psf-pixelscale', '1',
        '--output-file-format', 'FITS',
        '--output-file', output_catalog,
        '--output-properties', 'SourceIDs,PixelCentroid'
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
    assert np.isclose(table['pixel_centroid_x'][0], 82.2261)
    assert np.isclose(table['pixel_centroid_y'][0], 114.604)
