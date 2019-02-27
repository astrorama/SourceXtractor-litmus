import os

import numpy as np
import pytest
from astropy.table import Table

from util import stuff, plot


@pytest.fixture
def modelfitting_catalog(sextractorxx, datafiles, module_output_area, signal_to_noise_ratio):
    """
    Run sextractorxx on a single frame. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise, and sorted by location, so cross-matching is easier
    """
    sextractorxx.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,WorldCentroid,IsophotalFlux,FlexibleModelFitting',
            detection_image=datafiles / 'sim09' / 'img' / 'sim09_r_01.fits',
            python_config_file=datafiles / 'sim09' / 'sim09_single_modelfitting.py'
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= signal_to_noise_ratio
    catalog['flux_r_err'][catalog['flux_r_err'] >= 99.] = np.nan
    catalog['mag_r_err'][catalog['mag_r_err'] >= 99.] = np.nan
    return np.sort(catalog[bright_filter], order=('world_centroid_alpha', 'world_centroid_delta'))


def test_detection(modelfitting_catalog, reference):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(modelfitting_catalog) > 0
    assert len(modelfitting_catalog) == len(reference)


def test_magnitude(modelfitting_catalog, reference, stuff_simulation, tolerances):
    """
    Check the magnitude generated by the model fitting (a dependent column on the flux!)
    """
    stars, galaxies, kdtree = stuff_simulation
    expected_mags = np.append(stars.mag, galaxies.mag)

    target_closest = stuff.get_closest(modelfitting_catalog, kdtree)
    ref_closest = stuff.get_closest(reference, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY')

    target_mag = modelfitting_catalog['mag_r']
    ref_mag = reference['MAG_MODEL']
    mag_diff = np.abs((expected_mags[target_closest['source']] - target_mag))
    ref_diff = np.abs((expected_mags[ref_closest['source']] - ref_mag))

    assert np.median(mag_diff) <= np.median(ref_diff) * (1 + tolerances['magnitude'])


def test_generate_report(modelfitting_catalog, reference, stuff_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    with plot.Report(module_output_area / 'report.pdf') as report:
        loc_map = plot.Location(datafiles / 'sim09' / 'img' / 'sim09.fits', stuff_simulation)
        loc_map.add('SExtractor2', reference, 'ALPHA_SKY', 'DELTA_SKY', marker='1')
        loc_map.add('SExtractor++', modelfitting_catalog, 'world_centroid_alpha', 'world_centroid_delta', marker='3')
        report.add(loc_map)

        dist = plot.Distances(stuff_simulation)
        dist.add('SExtractor2', reference, 'ALPHA_SKY', 'DELTA_SKY', marker='o')
        dist.add('SExtractor++', modelfitting_catalog, 'world_centroid_alpha', 'world_centroid_delta', marker='.')
        report.add(dist)

        mag_r = plot.Magnitude('R', stuff_simulation)
        mag_r.add(
            'SExtractor2',
            reference, 'ALPHA_SKY', 'DELTA_SKY', 'MAG_MODEL', 'MAGERR_MODEL',
            marker='o'
        )
        mag_r.add(
            'SExtractor++',
            modelfitting_catalog, 'world_centroid_alpha', 'world_centroid_delta', 'mag_r', 'mag_r_err',
            marker='.'
        )
        report.add(mag_r)
