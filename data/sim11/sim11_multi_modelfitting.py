import os
import numpy as np

from glob import glob
from sourcextractor.config import *

args = Arguments(engine="levmar")
set_engine(args.engine)

# To match simulation (26 mag zeropoint, 300 exposure)
MAG_ZEROPOINT = 32.19

base_dir = os.path.abspath(os.path.dirname(__file__))

top = load_fits_images(
    sorted(glob(os.path.join(base_dir, 'img', 'sim11_[r|g]_*.fits.gz'))),
    sorted(glob(os.path.join(base_dir, 'psf', 'sim11_[r|g]_*.psf'))),
)
top.split(ByKeyword('FILTER'))

measurement_group = MeasurementGroup(top)

pixel_x, pixel_y = get_pos_parameters()
ratio = FreeParameter(1, Range((0, 10), RangeType.LINEAR))
rad = FreeParameter(lambda o: o.get_radius(), Range(lambda v, o: (.01 * v, 100 * v), RangeType.EXPONENTIAL))
angle = FreeParameter(lambda o: o.get_angle(), Range((-2 * 3.14159, 2 * 3.14159), RangeType.LINEAR))

for band, group in measurement_group:
    flux = get_flux_parameter()
    mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)
    add_model(group, ExponentialModel(pixel_x, pixel_y, flux, rad, ratio, angle))

    add_output_column('model_flux_' + band, flux)
    add_output_column('model_mag_' + band, mag)

add_output_column('model_x', pixel_x)
add_output_column('model_y', pixel_y)
add_output_column('model_rad', rad)
add_output_column('model_angle', angle)

print_model_fitting_info(top)
print_output_columns()
