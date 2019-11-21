from pytest import raises


def test_load_fits_images(sourcextractor_py):
    """
    Must fail if we try to load an empty list of images
    """
    with raises(ValueError):
        sourcextractor_py.load_fits_images([])


def test_add_model_measure_group(sourcextractor_py, datafiles):
    """
    Must not be able to add a model on top of an ImageGroup.
    Only MeasureGroup must be accepted.
    """
    top = sourcextractor_py.load_fits_images([datafiles / 'simple' / 'saturated.fits.gz'])
    assert isinstance(top, sourcextractor_py.ImageGroup)

    x, y = sourcextractor_py.get_pos_parameters()
    flux = sourcextractor_py.get_flux_parameter()
    with raises(TypeError):
        sourcextractor_py.add_model(top, sourcextractor_py.PointSourceModel(x, y, flux))

    # Must not fail
    measurement_group = sourcextractor_py.MeasurementGroup(top)
    sourcextractor_py.add_model(measurement_group, sourcextractor_py.PointSourceModel(x, y, flux))


def test_missing_file_in_python(sourcextractor_py, datafiles):
    with raises(Exception):
        sourcextractor_py.load_fits_images([datafiles / 'ouch.fits.gz'])

    with raises(Exception):
        sourcextractor_py.load_fits_images(
            [datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz'],
            psf_filename=[datafiles / 'sim11' / 'psf' / 'ouch.psf'],
        )
