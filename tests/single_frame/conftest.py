import pytest
from astropy.table import Table

from util import stuff
from util.validation import CrossValidation


@pytest.fixture(scope='session')
def sim09_r_simulation(datafiles, simulation_mag_zeropoint, simulation_exposure):
    """
    Fixture for the original stuff simulation
    """
    return stuff.Simulation(datafiles / 'sim09' / 'sim09_r.list', simulation_mag_zeropoint, simulation_exposure)


@pytest.fixture(scope='session')
def sim09_r_01_reference(datafiles, tolerances):
    """
    Fixture for the ref catalog, filtered by signal/noise.
    """
    catalog = Table.read(datafiles / 'sim09' / 'ref' / 'sim09_r_01_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='session')
def sim09_r_01_cross(sim09_r_01_reference, sim09_r_simulation, datafiles, tolerances):
    cross = CrossValidation(
        datafiles / 'sim09' / 'img' / 'sim09_r_01.fits', sim09_r_simulation,
        max_dist=tolerances['distance']
    )
    return cross(sim09_r_01_reference['X_IMAGE'], sim09_r_01_reference['Y_IMAGE'])


@pytest.fixture(scope='session')
def sim09_r_reference(datafiles, tolerances):
    """
    Fixture for the ref catalog, filtered by signal/noise.
    """
    catalog = Table.read(datafiles / 'sim09' / 'ref' / 'sim09_r_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='session')
def sim09_r_cross(sim09_r_reference, sim09_r_simulation, datafiles, tolerances):
    cross = CrossValidation(
        datafiles / 'sim09' / 'img' / 'sim09_r.fits', sim09_r_simulation,
        max_dist=tolerances['distance']
    )
    return cross(sim09_r_reference['X_IMAGE'], sim09_r_reference['Y_IMAGE'])
