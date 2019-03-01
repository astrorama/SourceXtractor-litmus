import numpy as np
import pytest
from astropy.table import Table

from util import stuff


@pytest.fixture(scope='session')
def stuff_simulation(datafiles):
    """
    Fixture for the original stuff simulation
    """
    return stuff.Simulation(datafiles / 'sim09' / 'sim09_r.list')


@pytest.fixture(scope='session')
def reference(datafiles, signal_to_noise_ratio):
    """
    Fixture for the ref catalog, filtered by signal/noise and sorted by location
    """
    catalog = Table.read(datafiles / 'sim09' / 'ref' / 'sim09_r_01_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= signal_to_noise_ratio
    return np.sort(catalog[bright_filter], order=('ALPHA_SKY', 'DELTA_SKY'))


@pytest.fixture(scope='session')
def coadded_reference(datafiles, signal_to_noise_ratio):
    """
    Fixture for the ref catalog, filtered by signal/noise and sorted by location
    """
    catalog = Table.read(datafiles / 'sim09' / 'ref' / 'sim09_r_reference.fits')
    bright_filter = catalog['FLUX_ISO'] / catalog['FLUXERR_ISO'] >= signal_to_noise_ratio
    return np.sort(catalog[bright_filter], order=('ALPHA_SKY', 'DELTA_SKY'))
