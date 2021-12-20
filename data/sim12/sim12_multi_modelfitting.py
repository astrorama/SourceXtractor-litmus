import os
import numpy as np

from glob import glob
from sourcextractor.config import *

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

top = load_fits_images(
    sorted(glob(os.path.join(base_dir, 'img', 'sim12_[r|g]_*.fits.gz'))),
    sorted(glob(os.path.join(base_dir, 'psf', 'sim12_[r|g]_*.psf'))),
)
top.split(ByKeyword('FILTER'))

measurement_group = MeasurementGroup(top)

pixel_x, pixel_y = get_pos_parameters()
ratio = FreeParameter(1, Range((0, 10), RangeType.LINEAR))
rad = FreeParameter(lambda o: o.radius, Range(lambda v, o: (.01 * v, 100 * v), RangeType.EXPONENTIAL))
angle = FreeParameter(lambda o: o.angle, Range((-2 * 3.14159, 2 * 3.14159), RangeType.LINEAR))
sersic = FreeParameter(2.0, Range((1.0, 7.0), RangeType.LINEAR))

for band, group in measurement_group:
    flux = get_flux_parameter()
    mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)
    add_model(group, SersicModel(pixel_x, pixel_y, flux, rad, ratio, angle, sersic))

    add_output_column('model_flux_' + band, flux)
    add_output_column('model_mag_' + band, mag)

add_output_column('model_x', pixel_x)
add_output_column('model_y', pixel_y)
add_output_column('model_rad', rad)
add_output_column('model_angle', angle)
add_output_column('model_sersic', sersic)

print_model_fitting_info(top)
print_output_columns()
