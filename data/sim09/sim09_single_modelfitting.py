from sextractorxx.config import *
import os
import numpy as np

# To match simulation (26 mag zeropoint, 300 exposure)
MAG_ZEROPOINT = 32.19

base_dir = os.path.abspath(os.path.dirname(__file__))

measurement_img = load_fits_image(
    os.path.join(base_dir, 'img', 'sim09_r_01.fits'),
    psf_file=os.path.join(base_dir, 'psf', 'sim09_r_01.psf'),
)
measurement_group = MeasurementGroup(ImageGroup(images=[measurement_img]))

alpha, delta = get_pos_parameters()
total_flux = get_flux_parameter()

angle = FreeParameter(lambda o: o.get_angle(), Range((-2 * 3.14159, 2 * 3.14159), RangeType.LINEAR))
ratio = FreeParameter(1, Range((0, 10), RangeType.LINEAR))
rad = FreeParameter(lambda o: o.get_radius(), Range(lambda v, o: (.01 * v, 100 * v), RangeType.EXPONENTIAL))
bulge_disk = FreeParameter(.5, Range((0, 1), RangeType.LINEAR))
flux = DependentParameter(lambda f, r: f * r, total_flux, bulge_disk)
mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)

add_model(measurement_group, ExponentialModel(alpha, delta, flux, rad, ratio, angle))

add_output_column('alpha', alpha)
add_output_column('delta', delta)
add_output_column('flux_r', flux)
add_output_column('mag_r', mag)

print_model_fitting_info(measurement_group)
print_output_columns()
