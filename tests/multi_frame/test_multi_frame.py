import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.image import Image
from util.matching import CrossMatching, intersect


@pytest.fixture(scope='module')
def multi_frame_run(sourcextractor, datafiles, module_output_area, tolerances):
    """
    Run sourcextractor on multiple frames. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise.
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels,AperturePhotometry',
        detection_image=datafiles / 'sim12' / 'img' / 'sim12_r.fits.gz',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12_r.weight.fits.gz',
        weight_type='weight',
        weight_absolute=True,
        python_config_file=datafiles / 'sim12' / 'sim12_multiframe.py'
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    return SimpleNamespace(run=run, catalog=catalog[bright_filter])


@pytest.fixture(scope='module')
def multi_frame_catalog(multi_frame_run):
    return multi_frame_run.catalog


@pytest.fixture
def multi_frame_cross(multi_frame_catalog, sim12_r_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim12' / 'img' / 'sim12_r.fits.gz',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12_r.weight.fits.gz'
    )
    cross = CrossMatching(image, sim12_r_simulation, max_dist=tolerances['distance'])
    return cross(multi_frame_catalog['pixel_centroid_x'], multi_frame_catalog['pixel_centroid_y'])


def test_iso_flux(multi_frame_catalog, sim12_r_reference, multi_frame_cross, sim12_r_cross):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    ISO is measured on the detection frame
    """
    catalog_intersect, ref_intersect = intersect(multi_frame_cross, sim12_r_cross)
    catalog_hits = multi_frame_catalog[multi_frame_cross.all_catalog[catalog_intersect]]
    ref_hits = sim12_r_reference[sim12_r_cross.all_catalog[ref_intersect]]

    assert len(catalog_hits) == len(ref_hits)

    catalog_flux = catalog_hits['isophotal_flux']
    catalog_flux_err = catalog_hits['isophotal_flux_err']
    ref_flux = ref_hits['FLUX_ISO']
    ref_flux_err = ref_hits['FLUXERR_ISO']
    real_flux = sim12_r_cross.all_fluxes[ref_intersect]

    catalog_dist = np.sqrt((catalog_flux - real_flux) ** 2 / catalog_flux_err ** 2)
    ref_dist = np.sqrt((ref_flux - real_flux) ** 2 / ref_flux_err ** 2)

    assert np.median(catalog_dist - ref_dist) <= 1e-6


@pytest.mark.parametrize(
    'frame', range(3)
)
def test_auto_flux(frame, multi_frame_catalog, sim12_r_reference, multi_frame_cross, sim12_r_cross):
    """
    AUTO is measured on the measurement frames, so it is trickier. Need to run the test for each
    frame, and filter out sources that are on the boundary or outside.
    """
    catalog_intersect, ref_intersect = intersect(multi_frame_cross, sim12_r_cross)
    catalog_hits = multi_frame_catalog[multi_frame_cross.all_catalog[catalog_intersect]]
    ref_hits = sim12_r_reference[sim12_r_cross.all_catalog[ref_intersect]]

    inframe_filter = (catalog_hits['auto_flags'][:, frame] == 0)

    catalog_flux = catalog_hits['auto_flux'][:, frame][inframe_filter]
    catalog_flux_err = catalog_hits['auto_flux_err'][:, frame][inframe_filter]
    ref_flux = ref_hits['FLUX_AUTO'][inframe_filter]
    ref_flux_err = ref_hits['FLUXERR_AUTO'][inframe_filter]
    real_flux = sim12_r_cross.all_fluxes[ref_intersect[inframe_filter]]

    catalog_dist = np.sqrt((catalog_flux - real_flux) ** 2 / catalog_flux_err ** 2)
    ref_dist = np.sqrt((ref_flux - real_flux) ** 2 / ref_flux_err ** 2)

    assert np.median(catalog_dist - ref_dist) <= 1e-6


@pytest.mark.parametrize(
    ['frame', 'aper_idx'], itertools.product(range(3), [0, 1, 2])
)
def test_aper_flux(frame, aper_idx, multi_frame_catalog, sim12_r_reference, multi_frame_cross, sim12_r_cross):
    """
    APERTURE is measured on the measurement frames, so it is trickier. Need to run the test for each
    frame, and filter out sources that are on the boundary or outside.
    """
    catalog_intersect, ref_intersect = intersect(multi_frame_cross, sim12_r_cross)
    catalog_hits = multi_frame_catalog[multi_frame_cross.all_catalog[catalog_intersect]]
    ref_hits = sim12_r_reference[sim12_r_cross.all_catalog[ref_intersect]]

    inframe_filter = (catalog_hits['aperture_flags'][:, frame, aper_idx] == 0)

    catalog_flux = catalog_hits['aperture_flux'][:, frame, aper_idx][inframe_filter]
    catalog_flux_err = catalog_hits['aperture_flux_err'][:, frame, aper_idx][inframe_filter]
    ref_flux = ref_hits['FLUX_APER'][inframe_filter][:, aper_idx]
    ref_flux_err = ref_hits['FLUXERR_APER'][inframe_filter][:, aper_idx]
    real_flux = sim12_r_cross.all_fluxes[ref_intersect[inframe_filter]]

    catalog_dist = np.sqrt((catalog_flux - real_flux) ** 2 / catalog_flux_err ** 2)
    ref_dist = np.sqrt((ref_flux - real_flux) ** 2 / ref_flux_err ** 2)

    assert (catalog_dist > 0).all()
    assert np.median(catalog_dist - ref_dist) <= 1e-6


@pytest.mark.report
def test_generate_report(multi_frame_run, sim12_r_reference, sim12_r_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    image = plot.Image(
        datafiles / 'sim12' / 'img' / 'sim12_r.fits.gz',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12_r.weight.fits.gz'
    )
    multi_frame_catalog = multi_frame_run.catalog
    with plot.Report(module_output_area / 'report.pdf') as report:
        loc_map = plot.Location(image, sim12_r_simulation)
        loc_map.add('SExtractor2 (R)', sim12_r_reference, 'X_IMAGE', 'Y_IMAGE', 'ISOAREA_IMAGE', marker='1')
        loc_map.add('SourceXtractor++', multi_frame_catalog, 'pixel_centroid_x', 'pixel_centroid_y', 'area', marker='2')
        report.add(loc_map)

        dist = plot.Distances(image, sim12_r_simulation)
        dist.add('SExtractor2 (R)', sim12_r_reference, 'X_IMAGE', 'Y_IMAGE', marker='o')
        dist.add('SourceXtractor++', multi_frame_catalog, 'pixel_centroid_x', 'pixel_centroid_y', marker='.')
        report.add(dist)

        mag_r = plot.Magnitude(f'iso_mag', sim12_r_simulation)
        mag_r.add(
            'SExtractor2', sim12_r_reference,
            'ALPHA_SKY', 'DELTA_SKY',
            'MAG_ISO', 'MAGERR_ISO',
            marker='o'
        )
        mag_r.add(
            'SourceXtractor++', multi_frame_catalog,
            'world_centroid_alpha', 'world_centroid_delta',
            'isophotal_mag', 'isophotal_mag_err',
            marker='.'
        )
        report.add(mag_r)

        for i in range(3):
            mag_r = plot.Magnitude(f'auto_mag:{i}', sim12_r_simulation)
            mag_r.add(
                'SExtractor2', sim12_r_reference,
                'ALPHA_SKY', 'DELTA_SKY',
                'MAG_AUTO', 'MAGERR_AUTO',
                marker='o'
            )
            mag_r.add(
                'SourceXtractor++', multi_frame_catalog,
                'world_centroid_alpha', 'world_centroid_delta',
                f'auto_mag:{i}', f'auto_mag_err:{i}',
                marker='.'
            )
            report.add(mag_r)

        for i in range(3):
            flag_r = plot.Flags(image)
            flag_r.set_sextractor2(
                'SExtractor2', sim12_r_reference,
                'X_IMAGE', 'Y_IMAGE', 'FLAGS'
            )
            flag_r.set_sourcextractor(
                f'SourceXtractor++ auto_flags:{i}', multi_frame_catalog,
                'pixel_centroid_x', 'pixel_centroid_y', f'auto_flags:{i}'
            )
            report.add(flag_r)

        report.add(plot.RunResult(multi_frame_run.run))
