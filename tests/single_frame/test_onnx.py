import os

import numpy as np
import pytest
from astropy.table import Table

from util.image import Image
from util.matching import CrossMatching
from scipy.ndimage import zoom


@pytest.fixture(scope='module')
def onnx_run(sourcextractor, datafiles, module_output_area, tolerances):
    if not os.path.exists(module_output_area / 'output.fits'):
        sourcextractor.set_output_directory(module_output_area)

        properties = ['SourceIDs',
                      'PixelCentroid',
                      'WorldCentroid',
                      'IsophotalFlux',
                      'ONNX',
                      'Vignet']

        run = sourcextractor(
            output_properties=','.join(properties),
            detection_image=datafiles / 'sim12' / 'img' / 'sim12_r_01.fits.gz',
            psf_filename=datafiles / 'sim12' / 'psf' / 'sim12_r_01.psf',
            onnx_model=[datafiles / 'onnx' / 'is_gal.onnx', datafiles / 'onnx' / 'super_resolution_10.onnx'],
            vignet_size=224
        )
        assert run.exit_code == 0

        catalog = Table.read(sourcextractor.get_output_catalog())
    else:
        catalog = Table.read(module_output_area / 'output.fits')
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='module')
def is_true_galaxy(onnx_run, sim12_r_simulation, datafiles):
    matcher = CrossMatching(
        image=Image(datafiles / 'sim12' / 'img' / 'sim12_r_01.fits.gz'),
        simulation=sim12_r_simulation

    )
    matches = matcher(onnx_run['pixel_centroid_x'], onnx_run['pixel_centroid_y'])
    is_galaxy = np.full(len(onnx_run), fill_value=False, dtype=bool)
    is_galaxy[matches.galaxies_catalog] = True
    return is_galaxy


def mcc(classification, galaxies):
    """
    See https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    A coefficient of +1 represents a perfect prediction, 0 no better than random prediction and
    −1 indicates total disagreement between prediction and observation.
    However, if MCC equals neither −1, 0, or +1, it is not a reliable indicator of how similar
    a predictor is to random guessing because MCC is dependent on the dataset.
    """
    tp = np.sum(classification[galaxies] == 1)
    fn = np.sum(classification[galaxies] == 0)
    tn = np.sum(classification[~galaxies] == 0)
    fp = np.sum(classification[~galaxies] == 1)
    nom = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        den = 1.
    return nom / den


def test_classification(onnx_run, is_true_galaxy):
    """
    Tests the results of a bad (NN) classifier. The point is testing
    the output of sourcextractor++, not the classification!
    """
    assert 'ch.unige.astro.IsGalaxy.is_galaxy' in onnx_run.columns
    class_column = onnx_run['ch.unige.astro.IsGalaxy.is_galaxy']
    assert class_column.dtype.type == np.int32
    assert len(class_column.shape) == 1
    assert np.array_equal(np.unique(class_column), [0, 1])

    # The trained NN is so-so, but at least the output should be better than random
    actual_mcc = mcc(class_column, is_true_galaxy)
    assert actual_mcc >= 0.59  # Far enough from 0 :)


def test_super_resolution(onnx_run):
    """
    The model comes form https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016
    It up-scales the vignette using NN. The original NN was not trained on astronomy images, of course,
    so the result is also bad, but, again, the point is to test the integration works.
    """
    assert 'torch-jit-export.output' in onnx_run.columns
    upscale_column = onnx_run['torch-jit-export.output']
    vignet_column = onnx_run['vignet']

    assert upscale_column.dtype.type == np.float32
    assert upscale_column.shape[1:] == (1, 672, 672)
    assert vignet_column.shape[1:] == (224, 224)

    # This is a bit crude, but again, it gives a sense that the output is not garbage
    # We take the center of the vignet and the upscaled image (so we focus on the source),
    # downscale the upscaled image, and take the coordinate with the highest value
    # If they correspond, it should be a very peaky distribution with the mode at 0
    vignet_cutsize = 16
    upscale_cutsize = vignet_cutsize * 3
    vignet_slice = slice(224 // 2 - vignet_cutsize // 2, 224 // 2 + vignet_cutsize // 2)
    upscale_slice = slice(672 // 2 - upscale_cutsize // 2, 672 // 2 + upscale_cutsize // 2)

    differences = np.zeros(len(onnx_run))
    for i in range(len(onnx_run)):
        vignet = vignet_column[i, vignet_slice, vignet_slice]
        # Not sure if a bug or a convention problem with the model/catalog, but the axes are reversed on the output
        upscale = upscale_column[i, 0, upscale_slice, upscale_slice].T
        # Downscale
        uz = zoom(upscale, zoom=1 / 3)
        # Coordinate difference
        differences[i] = np.nanargmax(vignet) - np.nanargmax(uz)

    # Majority must be at 0
    zero_count = (differences == 0).sum()
    assert zero_count / len(differences) >= 0.69
