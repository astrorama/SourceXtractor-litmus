import logging
import os

import numpy as np
import pytest
from astropy.table import Table

from util import stuff, plot
from util.image import Image
from util.validation import CrossValidation


@pytest.fixture
def modelfitting_catalog(sourcextractor, datafiles, module_output_area, tolerances):
    """
    Run sourcextractor on multiple frames. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise, and sorted by location, so cross-matching is easier
    """
    sourcextractor.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sourcextractor(
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,IsophotalFlux,FlexibleModelFitting,SourceFlags',
            detection_image=datafiles / 'sim11' / 'img' / 'sim11.fits',
            weight_image=datafiles / 'sim11' / 'img' / 'sim11.weight.fits',
            weight_type='weight',
            weight_absolute=True,
            python_config_file=datafiles / 'sim11' / 'sim11_multi_modelfitting.py',
            thread_count=4
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    catalog['model_flux_r_err'][catalog['model_flux_r_err'] >= 99.] = np.nan
    catalog['model_mag_r_err'][catalog['model_mag_r_err'] >= 99.] = np.nan
    catalog['model_flux_g_err'][catalog['model_flux_g_err'] >= 99.] = np.nan
    catalog['model_mag_g_err'][catalog['model_mag_g_err'] >= 99.] = np.nan
    assert len(catalog)
    return catalog[bright_filter]


@pytest.fixture
def r_cross(modelfitting_catalog, sim11_r_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim11' / 'img' / 'sim11.fits',
        weight_image=datafiles / 'sim11' / 'img' / 'sim11.weight.fits'
    )
    cross = CrossValidation(image, sim11_r_simulation, max_dist=tolerances['distance'])
    return cross(modelfitting_catalog['pixel_centroid_x'], modelfitting_catalog['pixel_centroid_y'])


@pytest.fixture
def g_cross(modelfitting_catalog, sim11_g_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim11' / 'img' / 'sim11.fits',
        weight_image=datafiles / 'sim11' / 'img' / 'sim11.weight.fits'
    )
    cross = CrossValidation(image, sim11_g_simulation, max_dist=tolerances['distance'])
    return cross(modelfitting_catalog['pixel_centroid_x'], modelfitting_catalog['pixel_centroid_y'])


@pytest.mark.slow
def test_magnitude(modelfitting_catalog, r_cross, g_cross):
    """
    Check the magnitude generated by the model fitting (a dependent column on the flux!)
    """
    r_hits = modelfitting_catalog[r_cross.all_catalog]
    g_hits = modelfitting_catalog[g_cross.all_catalog]

    assert len(r_hits)
    assert len(g_hits)

    r_not_flagged = r_hits['fmf_flags'] == 0
    g_not_flagged = g_hits['fmf_flags'] == 0

    r_mag = r_hits['model_mag_r']
    g_mag = g_hits['model_mag_g']

    r_diff = r_mag[r_not_flagged] - r_cross.all_magnitudes[r_not_flagged]
    g_diff = g_mag[g_not_flagged] - g_cross.all_magnitudes[g_not_flagged]

    assert np.mean(r_diff) <= 0.11
    assert np.mean(g_diff) <= 0.11


@pytest.mark.slow
def test_generate_report(modelfitting_catalog, sim11_r_simulation, sim11_g_simulation,
                         sim11_r_reference, sim11_g_reference,
                         datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """

    # Filter not fitted sources
    not_flagged = modelfitting_catalog['fmf_flags'] == 0

    image = plot.Image(
        datafiles / 'sim11' / 'img' / 'sim11.fits', weight_image=datafiles / 'sim11' / 'img' / 'sim11.weight.fits'
    )
    image_r = plot.Image(
        datafiles / 'sim11' / 'img' / 'sim11_r.fits',
        datafiles / 'sim11' / 'img' / 'sim11_r.weight.fits'
    )
    image_g = plot.Image(
        datafiles / 'sim11' / 'img' / 'sim11_g.fits',
        weight_image=datafiles / 'sim11' / 'img' / 'sim11_g.weight.fits'
    )
    with plot.Report(module_output_area / 'report.pdf') as report:
        loc_map = plot.Location(image, sim11_r_simulation)
        loc_map.add('SourceXtractor++', modelfitting_catalog[not_flagged], 'model_x', 'model_y', marker='3')
        report.add(loc_map)

        dist_r = plot.Distances(image_r, sim11_r_simulation)
        dist_r.add('SExtractor2 (R)', sim11_r_reference, 'XMODEL_IMAGE', 'YMODEL_IMAGE', marker='o')
        report.add(dist_r)

        dist_g = plot.Distances(image_g, sim11_g_simulation)
        dist_g.add('SExtractor2 (G)', sim11_g_reference, 'XMODEL_IMAGE', 'YMODEL_IMAGE', marker='h')
        report.add(dist_g)

        dist = plot.Distances(image, sim11_g_simulation)
        dist.add('SourceXtractor++', modelfitting_catalog[not_flagged], 'model_x', 'model_y', marker='.')
        report.add(dist)

        mag_r = plot.Magnitude('R', sim11_r_simulation)
        mag_r.add(
            'SExtractor2',
            sim11_r_reference, 'ALPHA_SKY', 'DELTA_SKY', 'MAG_MODEL', 'MAGERR_MODEL',
            marker='o'
        )
        mag_r.add(
            'SourceXtractor++',
            modelfitting_catalog[not_flagged], 'world_centroid_alpha', 'world_centroid_delta', 'model_mag_r', 'model_mag_r_err',
            marker='.'
        )
        report.add(mag_r)

        mag_g = plot.Magnitude('G', sim11_g_simulation)
        mag_g.add(
            'SExtractor2',
            sim11_r_reference, 'ALPHA_SKY', 'DELTA_SKY', 'MAG_MODEL', 'MAGERR_MODEL',
            marker='o'
        )
        mag_g.add(
            'SourceXtractor++',
            modelfitting_catalog[not_flagged], 'world_centroid_alpha', 'world_centroid_delta', 'model_mag_r', 'model_mag_r_err',
            marker='.'
        )
        report.add(mag_g)

        flags = plot.Flags(image)
        flags.set_sourcextractor(
            'SourceXtractor++ fmf_flags', modelfitting_catalog,
            'pixel_centroid_x', 'pixel_centroid_y', f'fmf_flags'
        )
        report.add(flags)
