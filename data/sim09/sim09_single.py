from sextractorxx.config import *
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

measurement_img = load_fits_image(
    os.path.join(base_dir, 'sim09.fits'),
    weight_file=os.path.join(base_dir, 'sim09.weight.fits'),
    weight_type='weight'
)

add_output_column(
    'aperture',
    add_aperture_photometry(measurement_img, [5, 20, 100])
)

print_output_columns()
