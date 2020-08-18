from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from util import plot
from util.catalog import get_column
from util.image import Image
from util.matching import intersect, CrossMatching


@pytest.fixture(scope='module')
def compressed_run(sourcextractor, datafiles, module_output_area, tolerances):
    """
    Run sourcextractor on a coadded single frame stored on a compressed FITS file,
    using tile compression.
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid,AutoPhotometry,IsophotalFlux,ShapeParameters,SourceFlags,NDetectedPixels',
        detection_image=datafiles / 'sim12' / 'img' / 'sim12_r.compressed.fits',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12_r.weight.compressed.fits',
        weight_type='weight',
        weight_absolute=True
    )
    assert run.exit_code == 0

    catalog = Table.read(sourcextractor.get_output_catalog())
    bright_filter = catalog['isophotal_flux'] / catalog['isophotal_flux_err'] >= tolerances['signal_to_noise']
    return SimpleNamespace(run=run, catalog=catalog[bright_filter])


@pytest.fixture(scope='module')
def compressed_catalog(compressed_run):
    return compressed_run.catalog


@pytest.fixture(scope='module')
def compressed_frame_cross(compressed_catalog, sim12_r_simulation, datafiles, tolerances):
    image = Image(
        datafiles / 'sim12' / 'img' / 'sim12_r.compressed.fits',
        weight_image=datafiles / 'sim12' / 'img' / 'sim12_r.weight.compressed.fits'
    )
    cross = CrossMatching(image, sim12_r_simulation, max_dist=tolerances['distance'])
    return cross(compressed_catalog['pixel_centroid_x'], compressed_catalog['pixel_centroid_y'])


@pytest.mark.parametrize(
    ['flux_column', 'reference_flux_column'], [
        [['isophotal_flux', 'isophotal_flux_err'], ['FLUX_ISO', 'FLUXERR_ISO']],
        [['auto_flux', 'auto_flux_err'], ['FLUX_AUTO', 'FLUXERR_AUTO']],
    ]
)
def test_flux(compressed_catalog, sim12_r_reference, flux_column, reference_flux_column, compressed_frame_cross,
              sim12_r_cross):
    """
    Cross-validate the magnitude columns. The measured magnitudes should be at least as close
    to the truth as the ref catalog (within a tolerance).
    We use only the hits, and ignore the detections that are a miss.
    """
    catalog_intersect, ref_intersect = intersect(compressed_frame_cross, sim12_r_cross)
    catalog_hits = compressed_catalog[compressed_frame_cross.all_catalog[catalog_intersect]]
    ref_hits = sim12_r_reference[sim12_r_cross.all_catalog[ref_intersect]]

    assert len(catalog_hits) == len(ref_hits)

    catalog_flux = get_column(catalog_hits, flux_column[0])
    catalog_flux_err = get_column(catalog_hits, flux_column[1])
    ref_flux = get_column(ref_hits, reference_flux_column[0])
    ref_flux_err = get_column(ref_hits, reference_flux_column[1])
    real_flux = sim12_r_cross.all_fluxes[ref_intersect]

    catalog_dist = np.sqrt((catalog_flux - real_flux) ** 2 / catalog_flux_err ** 2)
    ref_dist = np.sqrt((ref_flux - real_flux) ** 2 / ref_flux_err ** 2)

    assert np.median(catalog_dist - ref_dist) <= 1e-1


@pytest.mark.report
def test_generate_report(compressed_run, sim12_r_reference, sim12_r_simulation, datafiles, module_output_area):
    """
    Not quite a test. Generate a PDF report to allow for better insights.
    """
    plot.generate_report(
        module_output_area / 'report.pdf', sim12_r_simulation,
        datafiles / 'sim12' / 'img' / 'sim12_r.compressed.fits',
        compressed_run.catalog, sim12_r_reference,
        weight_image=datafiles / 'sim12' / 'img' / 'sim12_r.weight.compressed.fits',
        run=compressed_run.run
    )
