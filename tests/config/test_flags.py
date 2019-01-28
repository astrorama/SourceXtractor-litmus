import pytest
import re


def test_help(sextractorxx):
    """
    Test the --help flag.
    The program must print the list of possible parameters and exit with 0.
    """
    run = sextractorxx.run('--help')
    assert run.exit_code == 0
    assert '--version' in run.stdout
    assert '--list-output-properties' in run.stdout


def test_version(sextractorxx):
    """
    Test the --version flag.
    The program must print the version and exit with 0.
    """
    run = sextractorxx.run('--version')
    assert run.exit_code == 0
    assert re.match('^SExtractorxx \d+\.\d+(\.\d+)?$', run.stdout) is not None


def test_list_properties(sextractorxx):
    """
    Test the --list-properties flag.
    The program must print the list of possible properties and exit with 0.
    """
    run = sextractorxx.run('--list-output-properties')
    assert run.exit_code == 0
    assert 'AutoPhotometry' in run.stdout
    assert 'SourceIDs' in run.stdout
    assert 'PixelCentroid' in run.stdout


def test_invalid(sextractorxx):
    """
    Test an invalid flag (i.e --this-is-not-a-valid-flag).
    The program must exist with an error.
    """
    run = sextractorxx.run('--this-is-not-a-valid-flag')
    assert run.exit_code > 0
    assert 'unrecognised' in run.stderr


@pytest.mark.regression
def test_psf_pixel_scale_missing(sextractorxx, datafiles):
    """
    Regression test: --psf-fwhm without --psf-pixelscale segfaulted
    """
    single_source_fits = datafiles / 'single_source.fits'

    run = sextractorxx.run(
        '--detection-image', single_source_fits,
        '--psf-fwhm', '2',
        '--output-file-format', 'ASCII'
    )
    assert run.exit_code > 0
