from sourcextractor.config import *
from glob import glob
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

frames = [os.path.join(base_dir, 'mef_r.fits.gz')]
psfs = [os.path.join(base_dir, 'mef_r.psf')]

measurement_group = MeasurementGroup(load_fits_images(frames, psfs))

all_apertures = []
for img in measurement_group:
    all_apertures.extend(add_aperture_photometry(img, [5, 10, 20]))

add_output_column('aperture', all_apertures)

print_output_columns()
