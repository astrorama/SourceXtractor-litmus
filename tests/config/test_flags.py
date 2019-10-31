import tempfile

import pytest
import re


def test_help(sourcextractor):
    """
    Test the --help flag.
    The program must print the list of possible parameters and exit with 0.
    """
    run = sourcextractor('--help')
    assert run.exit_code == 0
    assert '--version' in run.stdout
    assert '--list-output-properties' in run.stdout


def test_version(sourcextractor):
    """
    Test the --version flag.
    The program must print the version and exit with 0.
    """
    run = sourcextractor('--version')
    assert run.exit_code == 0
    assert re.match('^SourceXtractorPlusPlus \\d+\\.?\\d+(\\.\\d+)?$', run.stdout) is not None


def test_list_properties(sourcextractor):
    """
    Test the --list-properties flag.
    The program must print the list of possible properties and exit with 0.
    """
    run = sourcextractor('--list-output-properties')
    assert run.exit_code == 0
    assert 'AutoPhotometry' in run.stdout
    assert 'SourceIDs' in run.stdout
    assert 'PixelCentroid' in run.stdout


def test_invalid(sourcextractor):
    """
    Test an invalid flag (i.e --this-is-not-a-valid-flag).
    The program must exist with an error.
    """
    run = sourcextractor('--this-is-not-a-valid-flag')
    assert run.exit_code > 0
    assert 'unrecognised' in run.stderr


def test_missing_detection_image(sourcextractor):
    """
    Pass a detection image that does not exist.
    """
    run = sourcextractor(
        detection_image='/tmp/does/surely/not/exist.fits'
    )
    assert run.exit_code > 0


def test_malformed_detection_image(sourcextractor):
    """
    Try opining a malformed image
    """
    malformed = tempfile.NamedTemporaryFile()
    malformed.write(b'\0')
    malformed.flush()
    run = sourcextractor(
        detection_image=malformed.name
    )
    assert run.exit_code > 0


def test_bad_segmentation_algorithm(sourcextractor):
    """
    Pass a bad segmentation algorithm
    """
    run = sourcextractor(
        segmentation_algorithm='UNKNOWN'
    )
    assert run.exit_code > 0


def test_bad_segmentation_filter(sourcextractor):
    """
    Pass a bad segmentation filter
    """
    malformed = tempfile.NamedTemporaryFile()
    malformed.write(b'abcdef')
    malformed.flush()
    run = sourcextractor(
        segmentation_filter=malformed.name
    )
    assert run.exit_code > 0


def test_bad_grouping_algorithm(sourcextractor):
    """
    Pass an invalid grouping algorithm
    """
    run = sourcextractor(
        grouping_algorihtm='UNKNOWN'
    )
    assert run.exit_code > 0
