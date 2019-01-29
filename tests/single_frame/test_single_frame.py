import numpy as np
import pytest
from util import stuff
from astropy.table import Table


@pytest.fixture
def single_frame(sextractorxx, datafiles):
    """
    Run sextractorxx on a single frame.
    """
    run = sextractorxx(
        output_properties='SourceIDs,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags',
        detection_image=datafiles / 'sim09' / 'sim09.fits',
        weight_image=datafiles / 'sim09' / 'sim09.weight.fits',
        weight_type='weight'
    )
    assert run.exit_code == 0

    catalog = Table.read(sextractorxx.get_output_catalog())

    stars, galaxies = stuff.parse_stuff_list(datafiles / 'sim09' / 'sim09.list')
    kdtree, _, _ = stuff.index_sources(stars, galaxies)

    closest = stuff.get_closest(catalog, kdtree)

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
    assert 500 < len(single_frame['output']['world_centroid_alpha']) < 2000


def test_location(single_frame):
    """
    Cross-validate the coordinates (alpha and delta) with the original suff simulation.
    From sextractor 2:
        Min:    7.648608605652947e-08
        Max:    0.004024216725538628
        Mean:   0.0002092625770859535
        StdDev: 0.0006004324668262142
        sum(squared): 0.0006537692269678584
    """
    distances = single_frame['distances']
    assert np.sum(distances ** 2) <= 0.0007


@pytest.mark.parametrize(
    ['flux_column', 'sum_squared_errors'], [
        ['auto_flux', 16400],
        ['isophotal_flux', 23000],
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
