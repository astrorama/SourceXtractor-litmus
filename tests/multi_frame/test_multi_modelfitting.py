import os

import numpy as np
import pytest
from astropy.table import Table

from util import stuff, plot


@pytest.fixture
def modelfitting_catalog(sextractorxx, datafiles, module_output_area, signal_to_noise_ratio):
    """
    Run sextractorxx on multiple frames. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise, and sorted by location, so cross-matching is easier
    """
    sextractorxx.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,WorldCentroid,IsophotalFlux,FlexibleModelFitting',
            detection_image=datafiles / 'sim09' / 'img' / 'sim09.fits',
            weight_image=datafiles / 'sim09' / 'img' / 'sim09.weight.fits',
            weight_type='weight',
            weight_absolute=True,
            python_config_file=datafiles / 'sim09' / 'sim09_multi_modelfitting.py',
            threads_nb=4
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= signal_to_noise_ratio
    catalog['model_flux_r_err'][catalog['model_flux_r_err'] >= 99.] = np.nan
    catalog['model_mag_r_err'][catalog['model_mag_r_err'] >= 99.] = np.nan
    catalog['model_flux_g_err'][catalog['model_flux_g_err'] >= 99.] = np.nan
    catalog['model_mag_g_err'][catalog['model_mag_g_err'] >= 99.] = np.nan
    return np.sort(catalog[bright_filter], order=('world_centroid_alpha', 'world_centroid_delta'))


@pytest.mark.slow
def test_magnitude(modelfitting_catalog, stuff_simulation_r, stuff_simulation_g, tolerances):
    """
    Check the magnitude generated by the model fitting (a dependent column on the flux!)
    """
    stars_r, galaxies_r, kdtree_r = stuff_simulation_r
    stars_g, galaxies_g, kdtree_g = stuff_simulation_g
    expected_mags_r = np.append(stars_r.mag, galaxies_r.mag)
    expected_mags_g = np.append(stars_g.mag, galaxies_g.mag)

    target_closest_r = stuff.get_closest(modelfitting_catalog, kdtree_r)
    target_closest_g = stuff.get_closest(modelfitting_catalog, kdtree_g)

    target_mag_r = modelfitting_catalog['model_mag_r']
    target_mag_g = modelfitting_catalog['model_mag_g']

    assert np.isclose(
        target_mag_r, expected_mags_r[target_closest_r['source']],
        rtol=tolerances['multiframe_magnitude']
    ).all()
    assert np.isclose(
        target_mag_g, expected_mags_g[target_closest_g['source']],
        rtol=tolerances['multiframe_magnitude']
    ).all()


@pytest.mark.slow
def test_generate_report(modelfitting_catalog, stuff_simulation_r, stuff_simulation_g, reference_r, reference_g,
                         datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    image = datafiles / 'sim09' / 'img' / 'sim09.fits'
    image_r =datafiles / 'sim09' / 'img' / 'sim09_r.fits'
    image_g = datafiles / 'sim09' / 'img' / 'sim09_g.fits'
    with plot.Report(module_output_area / 'report.pdf') as report:
        loc_map = plot.Location(image, stuff_simulation_r)
        loc_map.add('SExtractor++', modelfitting_catalog, 'model_x', 'model_y', marker='3')
        report.add(loc_map)

        dist_r = plot.Distances(image_r, stuff_simulation_r)
        dist_r.add('SExtractor2 (R)', reference_r, 'XMODEL_IMAGE', 'YMODEL_IMAGE', marker='o')
        report.add(dist_r)

        dist_g = plot.Distances(image_g, stuff_simulation_g)
        dist_g.add('SExtractor2 (G)', reference_g, 'XMODEL_IMAGE', 'YMODEL_IMAGE', marker='h')
        report.add(dist_g)

        dist = plot.Distances(image, stuff_simulation_g)
        dist.add('SExtractor++', modelfitting_catalog, 'model_x', 'model_y', marker='.')
        report.add(dist)

        mag_r = plot.Magnitude('R', stuff_simulation_r)
        mag_r.add(
            'SExtractor2',
            reference_r, 'ALPHA_SKY', 'DELTA_SKY', 'MAG_MODEL', 'MAGERR_MODEL',
            marker='o'
        )
        mag_r.add(
            'SExtractor++',
            modelfitting_catalog, 'world_centroid_alpha', 'world_centroid_delta', 'model_mag_r', 'model_mag_r_err',
            marker='.'
        )
        report.add(mag_r)

        mag_g = plot.Magnitude('G', stuff_simulation_g)
        mag_g.add(
            'SExtractor2',
            reference_r, 'ALPHA_SKY', 'DELTA_SKY', 'MAG_MODEL', 'MAGERR_MODEL',
            marker='o'
        )
        mag_g.add(
            'SExtractor++',
            modelfitting_catalog, 'world_centroid_alpha', 'world_centroid_delta', 'model_mag_r', 'model_mag_r_err',
            marker='.'
        )
        report.add(mag_g)
