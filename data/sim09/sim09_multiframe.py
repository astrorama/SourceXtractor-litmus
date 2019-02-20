from sextractorxx.config import *
from glob import glob
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

frames = glob(os.path.join(base_dir, 'img', 'sim09_r_*.fits'))
psfs = glob(os.path.join(base_dir, 'psf', 'sim09_r_*.psf'))

measurement_group = MeasurementGroup(load_fits_images(frames, psfs))

all_apertures = []
for img in measurement_group:
    all_apertures.extend(add_aperture_photometry(img, [5, 10, 20]))

add_output_column('aperture', all_apertures)

print_output_columns()
