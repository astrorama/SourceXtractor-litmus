import os

import numpy as np
from sourcextractor.config import *

base_dir = os.path.abspath(os.path.dirname(__file__))


def parse_flag(*args):
    if not args:
        return False
    return args[0].lower() == 'true'


args = Arguments(engine="levmar", iterative=parse_flag)
set_engine(args.engine)
use_iterative_fitting(args.iterative)

# Note that the hdu numbers need to be 1 higher for the compressed images!
top = ImageGroup(images=[
    MeasurementImage(
        fits_file=os.path.join(base_dir, 'des_compressed.fits.fz'),
        weight_file=os.path.join(base_dir, 'des_compressed.fits.fz'),
        psf_file=os.path.join(base_dir, 'des_psf.fits'),
        image_hdu=1,
        weight_hdu=2,
        psf_hdu=1,
        weight_type='weight',
        weight_absolute=1
    )]
)

mesgroup = MeasurementGroup(top)
set_max_iterations(250)
set_engine('levmar')
MAG_ZEROPOINT = 30.0
x, y = get_pos_parameters()
ra, dec = get_world_position_parameters(x, y)
flux = get_flux_parameter()
mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)
mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)
add_model(mesgroup, PointSourceModel(x, y, flux))
add_output_column('x', x)
add_output_column('y', y)
add_output_column('ra', ra)
add_output_column('dec', dec)
add_output_column('mag', mag)
add_output_column('flux_psf', flux)
