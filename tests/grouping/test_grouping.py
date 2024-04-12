import numpy as np
import pytest
from astropy.table import Table


def test_grouping_none(sourcextractor, datafiles, module_output_area):
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,GroupInfo',
        grouping_algorithm='NONE',
        partition_multithreshold=True,
        detection_minimum_area=20,
        detection_image=datafiles / 'grouping' / 'grouping.fits.gz',
        log_file='sourcextractor_test.log'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    # catalog should have 7 rows
    assert len(catalog) == 7

    # group_id should have 7 unique values
    assert len(np.unique(catalog['group_id'])) == 7


def test_grouping_split(sourcextractor, datafiles, module_output_area):
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,GroupInfo',
        grouping_algorithm='SPLIT',
        partition_multithreshold=True,
        detection_minimum_area=20,
        detection_image=datafiles / 'grouping' / 'grouping.fits.gz',
        log_file='sourcextractor_test.log'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    # catalog should have 7 rows
    assert len(catalog) == 7

    # group_id should have 3 unique values
    assert len(np.unique(catalog['group_id'])) == 3


def test_grouping_hardlimit(sourcextractor, datafiles, module_output_area):
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,GroupInfo',
        grouping_algorithm='SPLIT',
        grouping_hard_limit=1,
        partition_multithreshold=True,
        detection_minimum_area=20,
        detection_image=datafiles / 'grouping' / 'grouping.fits.gz',
        log_file='sourcextractor_test.log'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    # catalog should have 7 rows
    assert len(catalog) == 7

    # group_id should have 7 unique values
    assert len(np.unique(catalog['group_id'])) == 7


def test_grouping_moffat(sourcextractor, datafiles, module_output_area):
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,GroupInfo',
        grouping_algorithm='MOFFAT',
        grouping_moffat_threshold=0.01,
        partition_multithreshold=True,
        detection_minimum_area=20,
        detection_image=datafiles / 'grouping' / 'grouping.fits.gz',
        log_file='sourcextractor_test.log'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    # catalog should have 7 rows
    assert len(catalog) == 7

    # group_id should have 2 unique values
    assert len(np.unique(catalog['group_id'])) == 2


