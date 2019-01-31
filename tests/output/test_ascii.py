import os
import numpy as np
from astropy.io import ascii


def test_ascii_output_stdout(sextractorxx, datafiles):
    """
    Run SExtractor asking for ASCII output on the standard output
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    assert os.path.exists(single_source_fits)

    run = sextractorxx(
        detection_image=single_source_fits,
        output_file_format='ASCII',
        output_file='',
        output_properties='SourceIDs,PixelCentroid'
    )
    assert run.exit_code == 0

    table = ascii.read(run.stdout)
    assert len(table) == 1
    # Source ID
    assert table[0][0] == 1
    # Coordinates
    assert np.isclose(table[0][2], 21.5819)
    assert np.isclose(table[0][3], 24.0353)


def test_ascii_output_file(sextractorxx, datafiles):
    """
    Run SExtractor asking for ASCII output on a file
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    assert os.path.exists(single_source_fits)

    output_catalog = sextractorxx.get_output_directory() / 'output.txt'

    run = sextractorxx(
        detection_image=single_source_fits,
        output_file_format='ASCII',
        output_file=output_catalog,
        output_properties='SourceIDs,PixelCentroid'
    )
    assert run.exit_code == 0
    assert os.path.exists(output_catalog)

    table = ascii.read(output_catalog)
    assert len(table) == 1
    # Source ID
    assert table[0][0] == 1
    # Coordinates
    assert np.isclose(table[0][2], 21.5819)
    assert np.isclose(table[0][3], 24.0353)


def test_output_exists(sextractorxx, datafiles):
    """
    The output file already exists
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    output_catalog = sextractorxx.get_output_directory() / 'output.txt'

    os.makedirs(sextractorxx.get_output_directory(), exist_ok=True)
    with open(output_catalog, 'w') as fd:
        fd.write('TOUCH')

    run = sextractorxx(
        detection_image=single_source_fits,
        output_file_format='ASCII',
        output_file=output_catalog,
        output_properties='SourceIDs,PixelCentroid'
    )
    assert run.exit_code == 0
    assert os.path.exists(output_catalog)

    table = ascii.read(output_catalog)
    assert len(table) == 1
    # Source ID
    assert table[0][0] == 1
    # Coordinates
    assert np.isclose(table[0][2], 21.5819)
    assert np.isclose(table[0][3], 24.0353)
