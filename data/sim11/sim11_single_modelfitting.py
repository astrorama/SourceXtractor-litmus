from sourcextractor.config import *
import os
import numpy as np

args = Arguments(engine="levmar")
set_engine(args.engine)

# To match simulation (26 mag zeropoint, 300 exposure)
MAG_ZEROPOINT = 32.19

base_dir = os.path.abspath(os.path.dirname(__file__))

measurement_img = load_fits_image(
    os.path.join(base_dir, 'img', 'sim11_r_01.fits.gz'),
    psf=os.path.join(base_dir, 'psf', 'sim11_r_01.psf'),
)
measurement_group = MeasurementGroup(measurement_img)

pixel_x, pixel_y = get_pos_parameters()
flux = get_flux_parameter()

angle = FreeParameter(lambda o: o.get_angle(), Range((-2 * 3.14159, 2 * 3.14159), RangeType.LINEAR))
ratio = FreeParameter(1, Range((0, 10), RangeType.LINEAR))
rad = FreeParameter(lambda o: o.get_radius(), Range(lambda v, o: (.01 * v, 100 * v), RangeType.EXPONENTIAL))
mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)

add_model(measurement_group, ExponentialModel(pixel_x, pixel_y, flux, rad, ratio, angle))

add_output_column('model_x', pixel_x)
add_output_column('model_y', pixel_y)
add_output_column('model_flux_r', flux)
add_output_column('model_mag_r', mag)

print_model_fitting_info(measurement_group)
print_output_columns()
