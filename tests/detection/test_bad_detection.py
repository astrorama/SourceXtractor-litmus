def test_detection_no_fits(sourcextractor, datafiles):
    """
    Run with a file that exists, but is not a FITS file
    """
    not_a_fits = datafiles / 'sim11' / 'sim11_g.list'
    run = sourcextractor(
        detection_image=not_a_fits,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
    )
    assert run.exit_code == 1


def test_detection_no_image(sourcextractor, datafiles):
    """
    Run with a FITS file without any HDU image
    """
    not_image_hdu = datafiles / 'sim11' / 'ref' / 'sim11_g_reference.fits'
    run = sourcextractor(
        detection_image=not_image_hdu,
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
    )
    assert run.exit_code == 1
    assert 'non-image HDU' in run.stderr


def test_weight_no_fits(sourcextractor, datafiles):
    """
    Detection file that is not a FITS file
    """
    image_fits = datafiles / 'sim11' / 'img' / 'sim11.fits'
    not_a_fits = datafiles / 'sim11' / 'sim11_g.list'
    run = sourcextractor(
        detection_image=image_fits,
        weight_image=not_a_fits,
        weight_type='weight',
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
    )
    assert run.exit_code == 1


def test_weight_no_image(sourcextractor, datafiles):
    """
    Weight image that is a FITS without any HDU image
    """
    image_fits = datafiles / 'sim11' / 'img' / 'sim11.fits'
    not_image_hdu = datafiles / 'sim11' / 'ref' / 'sim11_g_reference.fits'
    run = sourcextractor(
        detection_image=image_fits,
        weight_image=not_image_hdu,
        weight_type='weight',
        output_properties='SourceIDs,PixelCentroid,SourceFlags',
    )
    assert run.exit_code == 1
    assert 'non-image HDU' in run.stderr
