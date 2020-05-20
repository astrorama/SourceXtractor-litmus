import pytest
from astropy.table import Table

from util import stuff
from util.image import Image
from util.matching import CrossMatching


@pytest.fixture(scope='session')
def sim11_r_simulation(datafiles, simulation_mag_zeropoint, simulation_exposure):
    """
    Fixture for the original stuff simulation
    """
    return stuff.Simulation(datafiles / 'sim11' / 'sim11_r.list', simulation_mag_zeropoint, simulation_exposure)


@pytest.fixture(scope='session')
def sim11_r_01_reference(datafiles, tolerances):
    """
    Fixture for the ref catalog, filtered by signal/noise.
    """
    catalog = Table.read(datafiles / 'sim11' / 'ref' / 'sim11_r_01_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='session')
def sim11_r_01_cross(sim11_r_01_reference, sim11_r_simulation, datafiles, tolerances):
    cross = CrossMatching(
        datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz', sim11_r_simulation,
        max_dist=tolerances['distance']
    )
    return cross(sim11_r_01_reference['X_IMAGE'], sim11_r_01_reference['Y_IMAGE'])


@pytest.fixture(scope='session')
def sim11_r_reference(datafiles, tolerances):
    """
    Fixture for the ref catalog, filtered by signal/noise.
    """
    catalog = Table.read(datafiles / 'sim11' / 'ref' / 'sim11_r_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='session')
def sim11_r_cross(sim11_r_reference, sim11_r_simulation, datafiles, tolerances):
    cross = CrossMatching(
        datafiles / 'sim11' / 'img' / 'sim11_r.fits.gz', sim11_r_simulation,
        max_dist=tolerances['distance']
    )
    return cross(sim11_r_reference['X_IMAGE'], sim11_r_reference['Y_IMAGE'])


@pytest.fixture
def coadded_frame_cross(coadded_catalog, sim11_r_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim11' / 'img' / 'sim11_r.fits.gz',
        weight_image=datafiles / 'sim11' / 'img' / 'sim11_r.weight.fits.gz'
    )
    cross = CrossMatching(image, sim11_r_simulation, max_dist=tolerances['distance'])
    return cross(coadded_catalog['pixel_centroid_x'], coadded_catalog['pixel_centroid_y'])
