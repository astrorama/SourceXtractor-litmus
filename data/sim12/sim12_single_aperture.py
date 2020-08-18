from sourcextractor.config import *
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

measurement_img = load_fits_image(
    os.path.join(base_dir, 'img', 'sim12_r_01.fits.gz'),
)

measurement_group = MeasurementGroup(measurement_img)

for img in measurement_group:
    add_output_column(
        'aperture',
        add_aperture_photometry(img, [5])
    )

print_output_columns()
