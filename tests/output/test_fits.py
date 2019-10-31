import os
import numpy as np
from astropy.io import fits


def test_fits_output_catalog_filename(sourcextractor, datafiles):
    """
    Run SExtractor asking for FITS output on a file
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    assert os.path.exists(single_source_fits)

    output_catalog = sourcextractor.get_output_directory() / 'output.fits'

    run = sourcextractor(
        detection_image=single_source_fits,
        output_catalog_format='FITS',
        output_catalog_filename=output_catalog,
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


def test_output_exists(sourcextractor, datafiles):
    """
    The output file already exists
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    output_catalog = sourcextractor.get_output_directory() / 'output.fits'

    os.makedirs(sourcextractor.get_output_directory(), exist_ok=True)
    with open(output_catalog, 'w') as fd:
        fd.write('TOUCH')

    run = sourcextractor(
        detection_image=single_source_fits,
        output_catalog_format='FITS',
        output_catalog_filename=output_catalog,
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
