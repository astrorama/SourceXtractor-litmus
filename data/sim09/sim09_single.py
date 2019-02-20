from sextractorxx.config import *
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

measurement_img = load_fits_image(
    os.path.join(base_dir, 'img', 'sim09_r_01.fits'),
)

add_output_column(
    'aperture',
    add_aperture_photometry(measurement_img, [5, 10, 20])
)

print_output_columns()
