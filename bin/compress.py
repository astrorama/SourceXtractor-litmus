#!/usr/bin/env python3
from astropy.io import fits
import os
import sys

for f in sys.argv[1:]:
    print(f)
    hdulist = fits.open(f)
    compressed_hdus = [fits.PrimaryHDU()]
    for hdu in hdulist:
        if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU) and hdu.data is not None and hdu.data.size:
            compressed = fits.CompImageHDU(hdu.data, hdu.header, quantize_level=32)
            compressed_hdus.append(compressed)
    name = os.path.basename(f)
    components = name.split('.')
    name = '.'.join([c for c in components if c not in ['fits', 'gz']])
    dest = os.path.join(os.path.dirname(f), name + '.compressed.fits')
    print(f'\t-> {dest}')
    fits.HDUList(compressed_hdus).writeto(dest, overwrite=True)
