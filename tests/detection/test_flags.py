import numpy as np
from astropy.table import Table

from util import stuff


def test_saturated_keyword(sourcextractor, datafiles):
    """
    Check if the saturated flag is set based on the image header when the configuration
    does not specify.
    """
    saturated_fits = datafiles / 'simple' / 'saturated.fits.gz'
    run = sourcextractor(
        detection_image=saturated_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        detection_image_saturation=None
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    assert len(catalog) == 1
    assert stuff.SourceFlags(catalog['source_flags']) is stuff.SourceFlags.SATURATED


def test_saturated_override(sourcextractor, datafiles):
    """
    Check if the saturated flag is *NOT* set when we override the detection image saturation
    """
    saturated_fits = datafiles / 'simple' / 'saturated.fits.gz'
    run = sourcextractor(
        detection_image=saturated_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        detection_image_saturation=100000
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    assert len(catalog) == 1
    assert stuff.SourceFlags(catalog['source_flags']) is stuff.SourceFlags.NONE


def test_boundary(sourcextractor, datafiles):
    """
    Check if the boundary flag is set on two of the three sources
    """
    saturated_fits = datafiles / 'simple' / 'boundary.fits.gz'
    run = sourcextractor(
        detection_image=saturated_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        detection_threshold=5
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    assert len(catalog) == 3
    assert np.sum(
        (catalog['source_flags'] & int(stuff.SourceFlags.BOUNDARY)).astype(bool)) == 2


def test_blended(sourcextractor, datafiles):
    """
    There are two de-blended sources
    """
    neighbours_fits = datafiles / 'simple' / 'neighbours.fits.gz'
    run = sourcextractor(
        detection_image=neighbours_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        detection_threshold=2,
        partition_multithreshold=True,
        detection_minimum_area=100
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())

    assert len(catalog) == 2
    assert np.sum(
        (catalog['source_flags'] & int(stuff.SourceFlags.BLENDED)).astype(bool)) == 2
