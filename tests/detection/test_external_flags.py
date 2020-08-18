import numpy as np
from astropy.table import Table


def test_external_missing_file(sourcextractor, datafiles):
    """
    Ask for external flags, but pass an invalid file.
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test='/tmp/does/not/exist.fits.gz',
        flag_type_test='OR'
    )
    assert run.exit_code > 0
    assert 'does not exist' in run.stderr


def test_external_bad_operator(sourcextractor, datafiles):
    """
    Ask for external flags, but pass an invalid operator.
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='XXX'
    )
    assert run.exit_code > 0
    assert 'Invalid option' in run.stderr


def test_external_bad_file(sourcextractor, datafiles):
    """
    Pass an invalid FITS file
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'sim12' / 'default.param',
        flag_type_test='OR'
    )
    assert run.exit_code > 0
    assert 'Can\'t open' in run.stderr


def test_external_bad_size(sourcextractor, datafiles):
    """
    Pass a valid FITS file but with a wrong size
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'boundary.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='OR'
    )
    assert run.exit_code != 0


def test_external_or(sourcextractor, datafiles):
    """
    Test OR
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='OR',
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    assert len(catalog) == 1

    source = catalog[0]

    # Flag 8 is outside the area where the source is
    # The area where the source is contains 4, 2 and 1
    assert source['isophotal_image_flags_test'] & 8 == 0
    assert source['isophotal_image_flags_test'] == 4 | 2 | 1  # 7


def test_external_and(sourcextractor, datafiles):
    """
    Test AND
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='AND',
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    assert len(catalog) == 1

    source = catalog[0]

    # Flag 8 is outside the area where the source is
    # The area where the source is contains 4, 2 and 1
    assert source['isophotal_image_flags_test'] & 8 == 0
    assert source['isophotal_image_flags_test'] == 4 & 2 & 1  # 0


def test_external_min(sourcextractor, datafiles):
    """
    Test MIN
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='MIN',
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    assert len(catalog) == 1

    source = catalog[0]

    # Flag 8 is outside the area where the source is
    # The area where the source is contains 4, 2 and 1
    assert source['isophotal_image_flags_test'] == 1


def test_external_max(sourcextractor, datafiles):
    """
    Test MAX
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='MAX',
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    assert len(catalog) == 1

    source = catalog[0]

    # Flag 8 is outside the area where the source is
    # The area where the source is contains 4, 2 and 1
    assert source['isophotal_image_flags_test'] == 4


def test_external_most(sourcextractor, datafiles):
    """
    Test MOST
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='MOST',
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    assert len(catalog) == 1

    source = catalog[0]

    # Flag 8 is outside the area where the source is
    # The area where the source is contains 4, 2 and 1. 2 is the most frequent.
    assert source['isophotal_image_flags_test'] == 2


def test_external_pass_two(sourcextractor, datafiles):
    """
    Pass two set of flag files
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'saturated.fits.gz',
        output_properties='SourceIDs,PixelCentroid,SourceFlags,ExternalFlags',
        flag_image_test=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test='OR',
        flag_image_test2=datafiles / 'simple' / 'saturated_flags.fits.gz',
        flag_type_test2='AND',
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    assert len(catalog) == 1

    source = catalog[0]

    # Flag 8 is outside the area where the source is
    assert source['isophotal_image_flags_test'] & 8 == 0
    assert source['isophotal_image_flags_test2'] & 8 == 0

    # The area where the source is contains 4, 2 and 1
    assert source['isophotal_image_flags_test'] == 4 | 2 | 1
    assert source['isophotal_image_flags_test2'] == 4 & 2 & 1
