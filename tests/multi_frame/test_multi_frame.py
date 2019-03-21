import itertools
import logging
import os

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.image import Image
from util.validation import CrossValidation


@pytest.fixture
def multi_frame_catalog(sextractorxx, datafiles, module_output_area, tolerances):
    """
    Run sextractorxx on multiple frames. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise.
    """
    sextractorxx.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels,AperturePhotometry',
            detection_image=datafiles / 'sim09' / 'img' / 'sim09_r.fits',
            weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits',
            weight_type='weight',
            weight_absolute=True,
            python_config_file=datafiles / 'sim09' / 'sim09_multiframe.py'
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    catalog['auto_mag'][catalog['auto_mag'] >= 99.] = np.nan
    catalog['aperture_mag'][catalog['aperture_mag'] >= 99.] = np.nan
    return catalog[bright_filter]


@pytest.fixture
def multi_frame_cross(multi_frame_catalog, sim09_r_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim09' / 'img' / 'sim09_r.fits',
        weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits'
    )
    cross = CrossValidation(image, sim09_r_simulation, max_dist=tolerances['distance'])
    return cross(multi_frame_catalog['pixel_centroid_x'], multi_frame_catalog['pixel_centroid_y'])


def test_detection(multi_frame_cross, sim09_r_cross):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(multi_frame_cross.stars_found) >= len(sim09_r_cross.stars_found)
    assert len(multi_frame_cross.galaxies_found) >= len(sim09_r_cross.galaxies_found)


def test_iso_magnitude(multi_frame_catalog, sim09_r_reference, multi_frame_cross, sim09_r_cross, tolerances):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    ISO is measured on the detection frame
    """
    catalog_hits = multi_frame_catalog[multi_frame_cross.all_catalog]
    ref_hits = sim09_r_reference[sim09_r_cross.all_catalog]

    catalog_mag = catalog_hits['isophotal_mag']
    ref_mag = ref_hits['MAG_ISO']

    catalog_mag_diff = catalog_mag - multi_frame_cross.all_magnitudes
    ref_mag_diff = ref_mag - sim09_r_cross.all_magnitudes

    assert np.median(catalog_mag_diff) <= np.median(ref_mag_diff) * (1 + tolerances['magnitude'])


@pytest.mark.parametrize(
    'frame', range(10)
)
def test_auto_magnitude(frame, multi_frame_catalog, sim09_r_reference, multi_frame_cross, sim09_r_cross, tolerances):
    """
    AUTO is measured on the measurement frames, so it is trickier. Need to run the test for each
    frame, and filter out sources that are on the boundary or outside.
    """
    catalog_hits = multi_frame_catalog[multi_frame_cross.all_catalog]
    ref_hits = sim09_r_reference[sim09_r_cross.all_catalog]

    target_filter = (np.isnan(catalog_hits['auto_mag'][:, frame]) == False)

    catalog_mag = catalog_hits['auto_mag'][:, frame]
    ref_mag = ref_hits['MAG_ISO']

    catalog_mag_diff = catalog_mag[target_filter] - multi_frame_cross.all_magnitudes[target_filter]
    ref_mag_diff = ref_mag - sim09_r_cross.all_magnitudes

    assert np.median(catalog_mag_diff) <= np.median(ref_mag_diff) * (1 + tolerances['magnitude'])


@pytest.mark.parametrize(
    ['frame', 'aper_idx'], itertools.product(range(10), [0, 1, 2])
)
def test_aper_magnitude(frame, aper_idx, multi_frame_catalog, sim09_r_reference, multi_frame_cross, sim09_r_cross,
                        tolerances):
    """
    APERTURE is measured on the measurement frames, so it is trickier. Need to run the test for each
    frame, and filter out sources that are on the boundary or outside.
    """
    catalog_hits = multi_frame_catalog[multi_frame_cross.all_catalog]
    ref_hits = sim09_r_reference[sim09_r_cross.all_catalog]

    target_filter = (np.isnan(catalog_hits['aperture_mag'][:, frame, aper_idx]) == False)

    catalog_mag = catalog_hits['aperture_mag'][:, frame, aper_idx]
    catalog_mag_err = catalog_hits['aperture_mag_err'][:, frame, aper_idx]
    ref_mag = ref_hits['MAG_APER'][:, aper_idx]
    ref_mag_err = ref_hits['MAGERR_APER'][:, aper_idx]

    catalog_mag_diff = catalog_mag[target_filter] - multi_frame_cross.all_magnitudes[target_filter]
    ref_mag_diff = ref_mag - sim09_r_cross.all_magnitudes

    catalog_chi2 = catalog_mag_diff ** 2 / catalog_mag_err[target_filter] ** 2
    ref_chi2 = ref_mag_diff ** 2 / ref_mag_err ** 2

    assert np.median(catalog_chi2) < np.median(ref_chi2)


def test_generate_report(multi_frame_catalog, sim09_r_reference, sim09_r_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    image = plot.Image(
        datafiles / 'sim09' / 'img' / 'sim09_r.fits',
        weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits'
    )
    with plot.Report(module_output_area / 'report.pdf') as report:
        loc_map = plot.Location(image, sim09_r_simulation)
        loc_map.add('SExtractor2 (R)', sim09_r_reference, 'X_IMAGE', 'Y_IMAGE', 'ISOAREA_IMAGE', marker='1')
        loc_map.add('SExtractor++', multi_frame_catalog, 'pixel_centroid_x', 'pixel_centroid_y', 'area', marker='2')
        report.add(loc_map)

        dist = plot.Distances(image, sim09_r_simulation)
        dist.add('SExtractor2 (R)', sim09_r_reference, 'X_IMAGE', 'Y_IMAGE', marker='o')
        dist.add('SExtractor++', multi_frame_catalog, 'pixel_centroid_x', 'pixel_centroid_y', marker='.')
        report.add(dist)

        for i in range(10):
            mag_r = plot.Magnitude(f'auto_mag:{i}', sim09_r_simulation)
            mag_r.add(
                'SExtractor2', sim09_r_reference,
                'ALPHA_SKY', 'DELTA_SKY',
                'MAG_AUTO', 'MAGERR_AUTO',
                marker='o'
            )
            mag_r.add(
                'SExtractor++', multi_frame_catalog,
                'world_centroid_alpha', 'world_centroid_delta',
                f'auto_mag:{i}', f'auto_mag_err:{i}',
                marker='.'
            )
            report.add(mag_r)

        for i in range(10):
            flag_r = plot.Flags(image)
            flag_r.set_sextractor2(
                'SExtractor2', sim09_r_reference,
                'X_IMAGE', 'Y_IMAGE', 'FLAGS'
            )
            flag_r.set_sextractorpp(
                f'SExtractor++ auto_flags:{i}', multi_frame_catalog,
                'pixel_centroid_x', 'pixel_centroid_y', f'auto_flags:{i}'
            )
            report.add(flag_r)
