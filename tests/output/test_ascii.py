import os
import numpy as np
from astropy.io import ascii


def test_ascii_output_stdout(sextractorxx, datafiles):
    """
    Run SExtractor asking for ASCII output on the standard output
    """
    single_source_fits = datafiles / 'single_source.fits'
    assert os.path.exists(single_source_fits)

    run = sextractorxx.run(
        '--detection-image', single_source_fits,
        '--psf-fwhm', '2', '--psf-pixelscale', '1',
        '--output-file-format', 'ASCII',
        '--output-properties', 'SourceIDs,PixelCentroid'
    )
    assert run.exit_code == 0

    table = ascii.read(run.stdout)
    assert len(table) == 1
    # Source ID
    assert table[0][0] == 1
    # Coordinates
    assert np.isclose(table[0][2], 82.2261)
    assert np.isclose(table[0][3], 114.604)


def test_ascii_output_file(sextractorxx, datafiles, output_directory):
    """
    Run SExtractor asking for ASCII output on a file
    """
    single_source_fits = datafiles / 'single_source.fits'
    assert os.path.exists(single_source_fits)

    output_catalog = output_directory / 'output.txt'

    run = sextractorxx.run(
        '--detection-image', single_source_fits,
        '--psf-fwhm', '2', '--psf-pixelscale', '1',
        '--output-file-format', 'ASCII',
        '--output-file', output_catalog,
        '--output-properties', 'SourceIDs,PixelCentroid'
    )
    assert run.exit_code == 0
    assert os.path.exists(output_catalog)

    table = ascii.read(output_catalog)
    assert len(table) == 1
    # Source ID
    assert table[0][0] == 1
    # Coordinates
    assert np.isclose(table[0][2], 82.2261)
    assert np.isclose(table[0][3], 114.604)
