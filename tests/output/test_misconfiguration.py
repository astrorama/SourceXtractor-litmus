from astropy.table import Table


def test_output_directory_missing(sextractorxx, datafiles):
    """
    The output directory does not exist.
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    output_catalog = sextractorxx.get_output_directory() / 'missing_dir' / 'output.fits'

    run = sextractorxx(
        detection_image=single_source_fits,
        output_catalog_filename_format='FITS',
        output_catalog_filename=output_catalog,
        output_properties='SourceIDs,PixelCentroid'
    )
    assert run.exit_code > 0


def test_wrong_output_property(sextractorxx, datafiles):
    """
    Ask for a property that does not exist
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    run = sextractorxx(
        detection_image=single_source_fits,
        output_catalog_filename_format='FITS',
        output_properties='SourceIDs,PixelCentroid,ThisIsAnInvalidPropertyIHope'
    )
    assert run.exit_code > 0


def test_no_output_properties(sextractorxx, datafiles):
    """
    No output properties configured. Still, it is expected to get a catalog
    with some default properties.
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    run = sextractorxx(
        detection_image=single_source_fits,
        output_properties=None
    )
    assert run.exit_code == 0

    table = Table.read(sextractorxx.get_output_catalog())
    assert len(table) == 1
