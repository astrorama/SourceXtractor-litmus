#
# This test is almost identical to test_single_frame but it does *NOT*
# use the measurement configuration file, and limits itself to measure
# the AutoPhotometry and IsophotalFlux on the detection image.
#
# The reason for this two separate runs is that the pre-release emitted different
# when the measurement frame matched the detection image, and when it was configured
# separately
#

import numpy as np
import pytest
from util import stuff
from astropy.table import Table


@pytest.fixture
def single_frame(sextractorxx, stuff_simulation, datafiles, module_output_area):
    """
    Run sextractorxx on a single frame. Overrides the output area per test so
    SExtractor is only run once for this setup
    """
    sextractorxx.set_output_directory(module_output_area)

    stars, galaxies, kdtree = stuff_simulation

    run = sextractorxx(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags',
        detection_image=datafiles / 'sim09' / 'sim09_r_01.fits',
    )
    assert run.exit_code == 0

    catalog = Table.read(sextractorxx.get_output_catalog())
    closest = stuff.get_closest(catalog, kdtree, alpha='pixel_centroid_x', delta='pixel_centroid_y')

    return {
        'output': catalog,
        'distances': closest['dist'],
        'expected_mags': np.append(stars.mag, galaxies.mag)[closest['source']]
    }


def test_detection(single_frame):
    """
    Quick test to verify there are objects detected.
    sextractor 2 detects 1473, but we can not really expect to have exactly the same.
    Just check there are a reasonable amount.
    """
    assert 50 < len(single_frame['output']['world_centroid_alpha']) < 1000


def test_location(single_frame):
    """
    Cross-validate the coordinates (X and Y for the single frame) with the original stuff simulation.
    Distance
        Min:    0.004604582201188226
        Max:    8.81317221359912
        Mean:   0.5255459265590932
        StdDev: 0.8319973288425666
        sum(squared): 244.04135518325353
    """
    distances = single_frame['distances']
    assert np.sum(distances ** 2) <= 245


@pytest.mark.parametrize(
    ['flux_column', 'sum_squared_errors'], [
        ['auto_flux', 46],
        ['isophotal_flux', 300],
    ]
)
def test_magnitude(single_frame, flux_column, sum_squared_errors, flux2mag):
    """
    Cross-validate flux columns
    """
    expected_mags = single_frame['expected_mags']
    fluxes = single_frame['output'][flux_column]

    expected_mags = expected_mags[fluxes > 0]
    fluxes = fluxes[fluxes > 0]

    mags = flux2mag(fluxes)

    diff = mags - expected_mags
    diff = diff[np.isnan(diff) == False]
    assert np.sum(diff ** 2) <= sum_squared_errors
