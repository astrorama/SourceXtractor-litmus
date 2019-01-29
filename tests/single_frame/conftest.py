import pytest
from util import stuff


@pytest.fixture(scope='session')
def stuff_simulation(datafiles):
    stars, galaxies = stuff.parse_stuff_list(datafiles / 'sim09' / 'sim09.list')
    kdtree, _, _ = stuff.index_sources(stars, galaxies)
    return stars, galaxies, kdtree
