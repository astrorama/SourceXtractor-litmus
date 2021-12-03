from sourcextractor.config import *
import os
import numpy as np

def parse_flag(*args):
    if not args:
        return False
    return args[0].lower() == 'true'

args = Arguments(engine="levmar", iterative=parse_flag)
set_engine(args.engine)
use_iterative_fitting(args.iterative)

# To match simulation (26 mag zeropoint, 300 exposure)
MAG_ZEROPOINT = 32.19

base_dir = os.path.abspath(os.path.dirname(__file__))

measurement_img = load_fits_image(
    os.path.join(base_dir, 'img', 'sim12_r_01.fits.gz'),
    psf=os.path.join(base_dir, 'psf', 'sim12_r_01.psf'),
)
measurement_group = MeasurementGroup(measurement_img)

pixel_x, pixel_y = get_pos_parameters()
ra,dec = get_world_position_parameters(pixel_x, pixel_y)
flux = get_flux_parameter()

angle = FreeParameter(lambda o: o.angle, Range((-2 * 3.14159, 2 * 3.14159), RangeType.LINEAR))
ratio = FreeParameter(1, Range((0, 10), RangeType.LINEAR))
rad = FreeParameter(lambda o: o.radius, Range(lambda v, o: (.01 * v, 100 * v), RangeType.EXPONENTIAL))
mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)
sersic = FreeParameter(2.0, Range((1.0, 7.0), RangeType.LINEAR))

add_prior(flux, lambda o: o.isophotal_flux, lambda o: 0.5 * o.isophotal_flux)
add_prior(sersic, 2, 0.5)

add_model(measurement_group, SersicModel(pixel_x, pixel_y, flux, rad, ratio, angle, sersic))

add_output_column('model_x', pixel_x)
add_output_column('model_y', pixel_y)
add_output_column('model_flux_r', flux)
add_output_column('model_mag_r', mag)
add_output_column('model_sersic', sersic)
add_output_column('model_ra', ra)
add_output_column('model_dec', dec)

print_model_fitting_info(measurement_group)
print_output_columns()
