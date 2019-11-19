from sourcextractor.config import *
import os

base_dir = os.path.abspath(os.path.dirname(__file__))

measurement_img = load_fits_image(
    os.path.join(base_dir, 'img', 'sim09_r.fits'),
    psf=os.path.join(base_dir, 'psf', 'sim09_r.psf'),
    weight=os.path.join(base_dir, 'img', 'sim09_r.weight.fits'),
    weight_type='weight',
    weight_absolute=True
)

measurement_group = MeasurementGroup(measurement_img)

for img in measurement_group:
    add_output_column(
        'aperture',
        add_aperture_photometry(img, [5, 10, 20])
    )

print_output_columns()
