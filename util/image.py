import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS


class Image(object):
    """
    Contains a FITS image, and its World Coordinate System, and an optional,
    associated, weight image.
    """

    def __init__(self, image, weight_image=None, hdu_index=0):
        """
        Constructor.
        :param image:
            Path to the FITS image.
        :param weight_image:
            Path to the FITS weight image. It can be None.
        :param hdu_index:
            Index of the HDU that contains the image data. Defaults to 0 (the primary one).
        """
        self.__image = fits.open(image)[hdu_index]
        self.__weight = fits.open(weight_image)[hdu_index] if weight_image else None
        self.__wcs = WCS(self.__image.header)

    @property
    def wcs(self):
        """
        :return:
            The World Coordinate System of the image.
        """
        return self.__wcs

    @property
    def data(self):
        """
        :return:
            numpy array with the data from the image data.
        """
        return self.__image.data

    @property
    def weight(self):
        """
        :return:
            numpy array with the data form the weight image, or None if there is no weight image.
        """
        return self.__weight.data if self.__weight is not None else None

    @property
    def size(self):
        """
        :return:
            Tuple with the size of the image data.
        """
        return self.__image.shape

    def for_display(self):
        """
        :return:
            Image data, massaged so it can be plotted nicely with matplotlib.
        """
        return ZScaleInterval()(self.data, clip=True)

    def get_contained_sources(self, ra, dec, **kwargs):
        """
        Filter sources that are outside of the image:
            1. It projects the right ascension and declination to pixel coordinates.
            2. Remove sources that are projected outside of the image.
            3. If there is a weight image, remove sources that correspond to areas with a weight of 0.
               (Useful for coadded images, for instance)
        :param ra:
            Right ascension
        :param dec:
            Declination
        :param kwargs:
            Arbitrary number of parameters, where each one is expected to be a numpy array with the same
            size of ra and dec. These parameters will also be filtered out.
        :return:
            A numpy record, with the x and y pixel coordinates for the sources that are within the image,
            and one extra element corresponding to each extra argument, using they keyword as name, also filtered.
        """
        assert len(ra) == len(dec)

        dtypes = [('x', float), ('y', float)]
        for k, v in kwargs.items():
            assert isinstance(v, np.ndarray)
            assert len(v) == len(ra)
            dtypes.append((k, v.dtype))

        h, w = self.size
        pix_coords = self.__wcs.all_world2pix(ra, dec, 1)
        pix_x = pix_coords[0]
        pix_y = pix_coords[1]
        inside_image = (pix_x >= 0) & (pix_x < w) & (pix_y >= 0) & (pix_y < h)

        # If we have a weight map, filter out those with weight 0
        if self.__weight is not None:
            weight_filter = self.weight[pix_y[inside_image].astype(np.int), pix_x[inside_image].astype(np.int)] != 0.
            inside_image[inside_image] = weight_filter

        nmatches = inside_image.sum()

        result = np.recarray((nmatches,), dtype=dtypes)
        result.x = pix_coords[0][inside_image]
        result.y = pix_coords[1][inside_image]
        for k, v in kwargs.items():
            result[k][:] = v[inside_image]

        return result
