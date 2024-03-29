import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.image import Image
from util.matching import CrossMatching

engines = ['levmar', 'gsl']
iterative = [False, True]
configurations = list(itertools.product(engines, iterative))
ids = [f'{e}_{"iterative" if i else "classic"}' for e, i in configurations]


@pytest.fixture(scope='module', params=configurations, ids=ids)
def modelfitting_run(request, sourcextractor, datafiles, module_output_area, tolerances):
    """
    Run sourcextractor on multiple frames. Overrides the output area per test so
    it is only run once for this setup.
    The output is filtered by signal/noise, and sorted by location, so cross-matching is easier
    """
    module_output_area = module_output_area / request.param[0]
    module_output_area /= 'iterative' if request.param[1] else 'classic'

    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        f'engine={request.param[0]}', f'iterative={request.param[1]}',
        grouping_algorithm='MOFFAT',
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,IsophotalFlux,FlexibleModelFitting,SourceFlags',
        detection_image=datafiles / 'sim12' / 'img' / 'sim12.fits.gz',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12.weight.fits.gz',
        weight_type='weight',
        weight_absolute=True,
        python_config_file=datafiles / 'sim12' / 'sim12_multi_modelfitting.py',
        thread_count=4
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    catalog.meta['output_area'] = module_output_area
    assert len(catalog)
    return SimpleNamespace(run=run, catalog=catalog[bright_filter])


@pytest.fixture(scope='module')
def modelfitting_catalog(modelfitting_run):
    return modelfitting_run.catalog


@pytest.fixture(scope='module')
def r_cross(modelfitting_catalog, sim12_r_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim12' / 'img' / 'sim12.fits.gz',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12.weight.fits.gz'
    )
    cross = CrossMatching(image, sim12_r_simulation, max_dist=tolerances['distance'])
    return cross(modelfitting_catalog['pixel_centroid_x'], modelfitting_catalog['pixel_centroid_y'])


@pytest.fixture(scope='module')
def g_cross(modelfitting_catalog, sim12_g_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim12' / 'img' / 'sim12.fits.gz',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12.weight.fits.gz'
    )
    cross = CrossMatching(image, sim12_g_simulation, max_dist=tolerances['distance'])
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

    r_diff = np.abs(r_mag[r_not_flagged] - r_cross.all_magnitudes[r_not_flagged])
    g_diff = np.abs(g_mag[g_not_flagged] - g_cross.all_magnitudes[g_not_flagged])

    assert np.nanmedian(r_diff) <= 0.1
    assert np.nanmedian(g_diff) <= 0.1


@pytest.mark.report
@pytest.mark.slow
def test_generate_report(modelfitting_run, sim12_r_simulation, sim12_g_simulation,
                         sim12_r_reference, sim12_g_reference,
                         datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    modelfitting_catalog = modelfitting_run.catalog
    module_output_area = modelfitting_catalog.meta['output_area']

    # Filter not fitted sources
    not_flagged = modelfitting_catalog['fmf_flags'] == 0

    image = plot.Image(
        datafiles / 'sim12' / 'img' / 'sim12.fits.gz', weight_image=datafiles / 'sim12' / 'img' / 'sim12.weight.fits.gz'
    )
    image_r = plot.Image(
        datafiles / 'sim12' / 'img' / 'sim12_r.fits.gz',
        datafiles / 'sim12' / 'img' / 'sim12_r.weight.fits.gz'
    )
    image_g = plot.Image(
        datafiles / 'sim12' / 'img' / 'sim12_g.fits.gz',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12_g.weight.fits.gz'
    )
    with plot.Report(module_output_area / 'report.pdf') as report:
        loc_map = plot.Location(image, sim12_r_simulation)
        loc_map.add('SourceXtractor++', modelfitting_catalog[not_flagged], 'model_x', 'model_y', marker='3')
        report.add(loc_map)

        dist_r = plot.Distances(image_r, sim12_r_simulation)
        dist_r.add('SExtractor2 (R)', sim12_r_reference, 'XMODEL_IMAGE', 'YMODEL_IMAGE', marker='o')
        report.add(dist_r)

        dist_g = plot.Distances(image_g, sim12_g_simulation)
        dist_g.add('SExtractor2 (G)', sim12_g_reference, 'XMODEL_IMAGE', 'YMODEL_IMAGE', marker='h')
        report.add(dist_g)

        dist = plot.Distances(image, sim12_g_simulation)
        dist.add('SourceXtractor++', modelfitting_catalog[not_flagged], 'model_x', 'model_y', marker='.')
        report.add(dist)

        mag_r = plot.Magnitude('R', sim12_r_simulation)
        mag_r.add(
            'SExtractor2',
            sim12_r_reference, 'ALPHA_SKY', 'DELTA_SKY', 'MAG_MODEL', 'MAGERR_MODEL',
            marker='o'
        )
        mag_r.add(
            'SourceXtractor++',
            modelfitting_catalog[not_flagged], 'world_centroid_alpha', 'world_centroid_delta', 'model_mag_r',
            'model_mag_r_err',
            marker='.'
        )
        report.add(mag_r)

        mag_g = plot.Magnitude('G', sim12_g_simulation)
        mag_g.add(
            'SExtractor2',
            sim12_r_reference, 'ALPHA_SKY', 'DELTA_SKY', 'MAG_MODEL', 'MAGERR_MODEL',
            marker='o'
        )
        mag_g.add(
            'SourceXtractor++',
            modelfitting_catalog[not_flagged], 'world_centroid_alpha', 'world_centroid_delta', 'model_mag_r',
            'model_mag_r_err',
            marker='.'
        )
        report.add(mag_g)

        flags = plot.Flags(image)
        flags.set_sourcextractor(
            'SourceXtractor++ fmf_flags', modelfitting_catalog,
            'pixel_centroid_x', 'pixel_centroid_y', f'fmf_flags'
        )
        report.add(flags)

        report.add(plot.RunResult(modelfitting_run.run))
