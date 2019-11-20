from astropy.table import Table


def test_missing_weight_image(sourcextractor, datafiles):
    """
    Pass a missing weight image
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'boundary.fits',
        weight_image='/tmp/missing/weight.fits',
        weight_type='weight'
    )
    assert run.exit_code > 0


def test_malformed_weight_image(sourcextractor, datafiles):
    """
    Pass a corrupted weight image
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'boundary.fits',
        weight_image=datafiles / 'sim11' / 'sim11_r.list',
        weight_type='weight'
    )
    assert run.exit_code > 0


def test_weight_bad_size(sourcextractor, datafiles):
    """
    Pass a weight image with a mismatching size
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'boundary.fits',
        weight_image=datafiles / 'simple' / 'saturated_flags.fits',
        weight_type='weight'
    )
    assert run.exit_code > 0


def test_weight_mask(sourcextractor, datafiles):
    """
    Pass a proper weight image, which filter the sources that are on the boundary
    """
    run = sourcextractor(
        detection_image=datafiles / 'simple' / 'boundary.fits',
        weight_image=datafiles / 'simple' / 'boundary_weights.fits',
        weight_type='weight',
        detection_threshold=5,
        output_properties='SourceIDs,PixelCentroid,SourceFlags'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    assert len(catalog) == 1

    assert catalog[0]['source_flags'] == 0
