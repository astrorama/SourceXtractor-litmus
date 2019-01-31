import numpy as np
from astropy.table import Table

from util import stuff


def test_saturated_keywork(sextractorxx, datafiles):
    """
    Check if the saturated flag is set based on the image header when the configuration
    does not specify.
    """
    saturated_fits = datafiles / 'simple' / 'saturated.fits'
    run = sextractorxx(
        detection_image=saturated_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        detection_image_saturation=None
    )
    assert run.exit_code == 0

    catalog = Table.read(sextractorxx.get_output_catalog())

    assert len(catalog) == 1
    assert stuff.SourceFlags(catalog['source_flags']) is stuff.SourceFlags.SATURATED


def test_saturated_override(sextractorxx, datafiles):
    """
    Check if the saturated flag is *NOT* set when we override the detection image saturation
    """
    saturated_fits = datafiles / 'simple' / 'saturated.fits'
    run = sextractorxx(
        detection_image=saturated_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        detection_image_saturation=100000
    )
    assert run.exit_code == 0

    catalog = Table.read(sextractorxx.get_output_catalog())

    assert len(catalog) == 1
    assert stuff.SourceFlags(catalog['source_flags']) is stuff.SourceFlags.NONE


def test_boundary(sextractorxx, datafiles):
    saturated_fits = datafiles / 'simple' / 'boundary.fits'
    run = sextractorxx(
        detection_image=saturated_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        threshold_value=5
    )
    assert run.exit_code == 0

    catalog = Table.read(sextractorxx.get_output_catalog())

    assert len(catalog) == 3
    assert np.sum(
        (catalog['source_flags'] & int(stuff.SourceFlags.BOUNDARY)).astype(np.bool)) == 2


def test_blended(sextractorxx, datafiles):
    neighbours_fits = datafiles / 'simple' / 'neighbours.fits'
    run = sextractorxx(
        detection_image=neighbours_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
        threshold_value=2,
        partition_multithreshold=True,
        detect_minarea=80
    )
    assert run.exit_code == 0

    catalog = Table.read(sextractorxx.get_output_catalog())

    assert len(catalog) == 2
    assert np.sum(
        (catalog['source_flags'] & int(stuff.SourceFlags.BLENDED)).astype(np.bool)) == 2
