import os


def test_output_directory_missing(sextractorxx, datafiles):
    """
    The output directory does not exist.
    """
    single_source_fits = datafiles / 'simple' / 'saturated.fits'
    output_catalog = sextractorxx.get_output_directory() / 'missing_dir' / 'output.fits'

    run = sextractorxx(
        detection_image=single_source_fits,
        output_file_format='FITS',
        output_file=output_catalog,
        output_properties='SourceIDs,PixelCentroid'
    )
    assert run.exit_code > 0
