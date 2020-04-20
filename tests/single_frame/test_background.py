import pytest


@pytest.mark.regression
@pytest.mark.parametrize(
    ['cell_size'], [[0], [32], [64], [768], [1536], [2048]]
)
def test_background_cellsize(sourcextractor, datafiles, module_output_area, cell_size):
    """
    Run sourcextractor with different cell sizes: from 0 (which should disable background
    modelling) to the full image, to more than the image.
    This is a regression test: cell sizes bigger than the image size caused a segfault
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid',
        detection_image=datafiles / 'sim11' / 'img' / 'sim11_r_01.fits.gz',
        background_cell_size=cell_size,
        log_file='sourcextractor_{}.log'.format(cell_size)
    )
    assert run.exit_code == 0
