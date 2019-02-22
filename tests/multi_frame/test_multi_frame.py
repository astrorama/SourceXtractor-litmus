import itertools
import os

import numpy as np
import pytest
from astropy.table import Table

from util import stuff, plot


@pytest.fixture
def multi_frame_catalog(sextractorxx, datafiles, module_output_area, signal_to_noise_ratio):
    """
    Run sextractorxx on multiple frames. Overrides the output area per test so
    SExtractor is only run once for this setup.
    The output is filtered by signal/noise, and sorted by location, so cross-matching is easier
    """
    sextractorxx.set_output_directory(module_output_area)

    output_catalog = module_output_area / 'output.fits'
    if not os.path.exists(output_catalog):
        run = sextractorxx(
            output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels',
            detection_image=datafiles / 'sim09' / 'img' / 'sim09_r.fits',
            weight_image=datafiles / 'sim09' / 'img' / 'sim09_r.weight.fits',
            weight_type='weight',
            weight_absolute=True,
            python_config_file=datafiles / 'sim09' / 'sim09_multiframe.py'
        )
        assert run.exit_code == 0

    catalog = Table.read(output_catalog)
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= signal_to_noise_ratio
    catalog['auto_mag'][catalog['auto_mag'] >= 99.] = np.nan
    catalog['aperture_mag'][catalog['aperture_mag'] >= 99.] = np.nan
    return np.sort(catalog[bright_filter], order=('world_centroid_alpha', 'world_centroid_delta'))


def test_detection(multi_frame_catalog, reference_r):
    """
    Test that the number of results matches the ref, and that they are reasonably close
    """
    assert len(multi_frame_catalog) > 0
    assert len(multi_frame_catalog) == len(reference_r)


def test_location(multi_frame_catalog, reference_r, stuff_simulation_r, tolerances):
    """
    The detections should be at least as close as the ref to the truth.
    Single frame simulations are in pixel coordinates.
    """
    _, _, kdtree = stuff_simulation_r

    det_closest = stuff.get_closest(
        multi_frame_catalog, kdtree
    )
    ref_closest = stuff.get_closest(
        reference_r, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY'
    )

    assert np.mean(det_closest['dist']) <= np.mean(ref_closest['dist']) * (1 + tolerances['distance'])


def test_iso_magnitude(multi_frame_catalog, reference_r, stuff_simulation_r, tolerances):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    ISO is measured on the detection frame
    """
    stars, galaxies, kdtree = stuff_simulation_r
    expected_mags = np.append(stars.mag, galaxies.mag)

    target_closest = stuff.get_closest(multi_frame_catalog, kdtree)
    ref_closest = stuff.get_closest(
        reference_r, kdtree, alpha='ALPHA_SKY', delta='DELTA_SKY'
    )

    target_mag = multi_frame_catalog['isophotal_mag']
    ref_mag = reference_r['MAG_ISO']

    relative_target_diff = np.abs((expected_mags[target_closest['source']] - target_mag) / target_mag)
    relative_ref_diff = np.abs((expected_mags[ref_closest['source']] - ref_mag) / ref_mag)

    assert np.median(relative_target_diff) <= np.median(relative_ref_diff) * (1 + tolerances['magnitude'])


@pytest.mark.parametrize(
    'frame', range(10)
)
def test_auto_magnitude(frame, multi_frame_catalog, stuff_simulation_r, tolerances):
    """
    AUTO is measured on the measurement frames, so it is trickier. Need to run the test for each
    frame, and filter out sources that are on the boundary or outside.
    """
    stars, galaxies, kdtree = stuff_simulation_r
    expected_mags = np.append(stars.mag, galaxies.mag)

    target_filter = (multi_frame_catalog['auto_flags'][:, frame] == 0)

    target = multi_frame_catalog[target_filter]
    target_closest = stuff.get_closest(target, kdtree)
    target_mag = target['auto_mag'][:, frame]

    assert np.isclose(
        target_mag, expected_mags[target_closest['source']], rtol=tolerances['multiframe_magnitude']
    ).all()


@pytest.mark.parametrize(
    ['frame', 'aper_idx'], itertools.product(range(10), [0, 1, 2])
)
def test_aper_magnitude(frame, aper_idx, multi_frame_catalog, stuff_simulation_r, tolerances):
    """
    APERTURE is measured on the measurement frames, so it is trickier. Need to run the test for each
    frame, and filter out sources that are on the boundary or outside.
    """
    stars, galaxies, kdtree = stuff_simulation_r
    expected_mags = np.append(stars.mag, galaxies.mag)

    target_filter = (multi_frame_catalog['aperture_flags'][:, frame, aper_idx] == 0)

    target = multi_frame_catalog[target_filter]
    target_closest = stuff.get_closest(target, kdtree)
    target_mag = target['aperture_mag'][:, frame, aper_idx]

    assert np.isclose(
        target_mag, expected_mags[target_closest['source']],
        rtol=tolerances['multiframe_magnitude']
    ).all()


def test_generate_report(multi_frame_catalog, reference_r, stuff_simulation_r, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    with plot.Report(module_output_area / 'report.pdf') as report:
        loc_map = plot.Location(datafiles / 'sim09' / 'img' / 'sim09.fits')
        loc_map.add('SExtractor2 (R)', reference_r, 'ALPHA_SKY', 'DELTA_SKY', marker='1')
        loc_map.add('SExtractor++', multi_frame_catalog, 'world_centroid_alpha', 'world_centroid_delta', marker='3')
        report.add(loc_map)

        dist = plot.Distances(stuff_simulation_r)
        dist.add('SExtractor2 (R)', reference_r, 'ALPHA_SKY', 'DELTA_SKY')
        dist.add('SExtractor++', multi_frame_catalog, 'world_centroid_alpha', 'world_centroid_delta')
        report.add(dist)

        for i in range(10):
            mag_r = plot.Magnitude(f'auto_mag:{i}', stuff_simulation_r)
            mag_r.add(
                'SExtractor2', reference_r,
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
            flag_r = plot.Flags(datafiles / 'sim09' / 'img' / 'sim09.fits')
            flag_r.set1(
                'SExtractor2', reference_r,
                'ALPHA_SKY', 'DELTA_SKY', 'FLAGS'
            )
            flag_r.set2(
                f'SExtractor++ auto_flags:{i}', multi_frame_catalog,
                'world_centroid_alpha', 'world_centroid_delta', f'auto_flags:{i}'
            )
            report.add(flag_r)
