import tempfile

import pytest
import re


def test_help(sextractorxx):
    """
    Test the --help flag.
    The program must print the list of possible parameters and exit with 0.
    """
    run = sextractorxx('--help')
    assert run.exit_code == 0
    assert '--version' in run.stdout
    assert '--list-output-properties' in run.stdout


def test_version(sextractorxx):
    """
    Test the --version flag.
    The program must print the version and exit with 0.
    """
    run = sextractorxx('--version')
    assert run.exit_code == 0
    assert re.match('^SExtractorxx \d+\.\d+(\.\d+)?$', run.stdout) is not None


def test_list_properties(sextractorxx):
    """
    Test the --list-properties flag.
    The program must print the list of possible properties and exit with 0.
    """
    run = sextractorxx('--list-output-properties')
    assert run.exit_code == 0
    assert 'AutoPhotometry' in run.stdout
    assert 'SourceIDs' in run.stdout
    assert 'PixelCentroid' in run.stdout


def test_invalid(sextractorxx):
    """
    Test an invalid flag (i.e --this-is-not-a-valid-flag).
    The program must exist with an error.
    """
    run = sextractorxx('--this-is-not-a-valid-flag')
    assert run.exit_code > 0
    assert 'unrecognised' in run.stderr


@pytest.mark.regression
def test_psf_pixel_scale_missing(sextractorxx, datafiles):
    """
    Regression test: --psf-fwhm without --psf-pixelscale segfaulted
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'

    run = sextractorxx(
        detection_image=single_source_fits,
        psf_fwhm='2', psf_pixelscale=None,
    )
    assert run.exit_code > 0


def test_missing_detection_image(sextractorxx):
    """
    Pass a detection image that does not exist.
    """
    run = sextractorxx(
        detection_image='/tmp/does/surely/not/exist.fits'
    )
    assert run.exit_code > 0


def test_malformed_detection_image(sextractorxx):
    """
    Try opining a malformed image
    """
    malformed = tempfile.NamedTemporaryFile()
    malformed.write(b'\0')
    malformed.flush()
    run = sextractorxx(
        detection_image=malformed.name
    )
    assert run.exit_code > 0


def test_bad_segmentation_algorithm(sextractorxx):
    """
    Pass a bad segmentation algorithm
    """
    run = sextractorxx(
        segmentation_algorithm='UNKNOWN'
    )
    assert run.exit_code > 0


def test_bad_segmentation_filter(sextractorxx):
    """
    Pass a bad segmentation filter
    """
    malformed = tempfile.NamedTemporaryFile()
    malformed.write(b'abcdef')
    malformed.flush()
    run = sextractorxx(
        segmentation_filter=malformed.name
    )
    assert run.exit_code > 0


def test_bad_grouping_algorithm(sextractorxx):
    """
    Pass an invalid grouping algorithm
    """
    run = sextractorxx(
        grouping_algorihtm='UNKNOWN'
    )
    assert run.exit_code > 0
