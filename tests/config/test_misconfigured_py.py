from pytest import raises


def test_load_fits_images(sextractorxx_py):
    """
    Must fail if we try to load an empty list of images
    """
    with raises(ValueError):
        sextractorxx_py.load_fits_images([])


def test_add_model_measure_group(sextractorxx_py, datafiles):
    """
    Must not be able to add a model on top of an ImageGroup.
    Only MeasureGroup must be accepted.
    """
    top = sextractorxx_py.load_fits_images([datafiles / 'simple' / 'saturated.fits'])
    assert isinstance(top, sextractorxx_py.ImageGroup)

    x, y = sextractorxx_py.get_pos_parameters()
    flux = sextractorxx_py.get_flux_parameter()
    with raises(TypeError):
        sextractorxx_py.add_model(top, sextractorxx_py.PointSourceModel(x, y, flux))

    # Must not fail
    measurement_group = sextractorxx_py.MeasurementGroup(top)
    sextractorxx_py.add_model(measurement_group, sextractorxx_py.PointSourceModel(x, y, flux))
