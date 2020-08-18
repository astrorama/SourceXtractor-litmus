import pytest
from astropy.table import Table

from util import stuff
from util.matching import CrossMatching


@pytest.fixture(scope='session')
def sim12_r_simulation(datafiles, simulation_mag_zeropoint, simulation_exposure):
    """
    Fixture for the original stuff simulation
    """
    return stuff.Simulation(datafiles / 'sim12' / 'sim12_r.list', simulation_mag_zeropoint, simulation_exposure)


@pytest.fixture(scope='session')
def sim12_g_simulation(datafiles, simulation_mag_zeropoint, simulation_exposure):
    """
    Fixture for the original stuff simulation
    """
    return stuff.Simulation(datafiles / 'sim12' / 'sim12_g.list', simulation_mag_zeropoint, simulation_exposure)


@pytest.fixture(scope='session')
def sim12_r_reference(datafiles, tolerances):
    """
    Fixture for the ref catalog, filtered by signal/noise.
    """
    catalog = Table.read(datafiles / 'sim12' / 'ref' / 'sim12_r_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='session')
def sim12_g_reference(datafiles, tolerances):
    """
    Fixture for the ref catalog, filtered by signal/noise and sorted by location
    """
    catalog = Table.read(datafiles / 'sim12' / 'ref' / 'sim12_g_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= tolerances['signal_to_noise']
    return catalog[bright_filter]


@pytest.fixture(scope='session')
def sim12_r_cross(sim12_r_reference, sim12_r_simulation, datafiles, tolerances):
    cross = CrossMatching(
        datafiles / 'sim12' / 'img' / 'sim12_r.fits.gz', sim12_r_simulation,
        max_dist=tolerances['distance']
    )
    return cross(sim12_r_reference['X_IMAGE'], sim12_r_reference['Y_IMAGE'])
