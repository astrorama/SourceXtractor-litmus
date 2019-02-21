import os
import numpy as np

from glob import glob
from sextractorxx.config import *

# To match simulation (26 mag zeropoint, 300 exposure)
MAG_ZEROPOINT = 32.19

base_dir = os.path.abspath(os.path.dirname(__file__))

top = load_fits_images(
    sorted(glob(os.path.join(base_dir, 'img', 'sim09_[r|g]_*.fits'))),
    sorted(glob(os.path.join(base_dir, 'psf', 'sim09_[r|g]_*.psf'))),
)
top.split(ByKeyword('FILTER'))

measurement_group = MeasurementGroup(top)

alpha, delta = get_pos_parameters()
ratio = FreeParameter(1, Range((0, 10), RangeType.LINEAR))
rad = FreeParameter(lambda o: o.get_radius(), Range(lambda v, o: (.01 * v, 100 * v), RangeType.EXPONENTIAL))
angle = FreeParameter(lambda o: o.get_angle(), Range((-2 * 3.14159, 2 * 3.14159), RangeType.LINEAR))

iso_flux = get_flux_parameter()

for band, group in top:
    bulge_disk = FreeParameter(.5, Range((0, 1), RangeType.LINEAR))
    flux = DependentParameter(lambda f, r: f * r, iso_flux, bulge_disk)
    mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)
    add_model(group, ExponentialModel(alpha, delta, flux, rad, ratio, angle))

    add_output_column('flux_' + band, flux)
    add_output_column('mag_' + band, mag)
    add_output_column('bulge_' + band, bulge_disk)

add_output_column('alpha', alpha)
add_output_column('delta', delta)
add_output_column('rad', rad)
add_output_column('angle', angle)

print_model_fitting_info(top)
print_output_columns()
