from sourcextractor.config import *
from glob import glob
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

cube = os.path.join(base_dir, 'cube.fits.gz')

measurement_group = MeasurementGroup(load_fits_data_cube(cube))

all_apertures = []
for img in measurement_group:
    all_apertures.extend(add_aperture_photometry(img, [5, 10, 20]))

add_output_column('aperture', all_apertures)

print_output_columns()
