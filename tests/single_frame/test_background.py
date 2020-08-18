import pytest


@pytest.mark.regression
@pytest.mark.parametrize(
    ['cell_size'], [[32], [64], [768], [1536], [2048]]
)
def test_background_cellsize(sourcextractor, datafiles, module_output_area, cell_size):
    """
    Run sourcextractor with different cell sizes: up to the full image, and more than the image.
    This is a regression test: cell sizes bigger than the image size caused a segfault
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid',
        detection_image=datafiles / 'sim12' / 'img' / 'sim12_r_01.fits.gz',
        background_cell_size=cell_size,
        log_file='sourcextractor_{}.log'.format(cell_size)
    )
    assert run.exit_code == 0


@pytest.mark.regression
@pytest.mark.parametrize(
    ['cell_size'], [[0], [-1]]
)
def test_background_bad_cellsize(sourcextractor, datafiles, module_output_area, cell_size):
    """
    If the cell size is <= 0, then sourcextractor is expected to fail with a descriptive error
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid',
        detection_image=datafiles / 'sim12' / 'img' / 'sim12_r_01.fits.gz',
        background_cell_size=cell_size,
        log_file='sourcextractor_{}.log'.format(cell_size)
    )
    assert run.exit_code == 1
    assert 'There are value(s) < 1 in backgound-cell-size: {}'.format(cell_size) in run.stderr


@pytest.mark.regression
@pytest.mark.parametrize(
    ['smoothing_box'], [[0], [1], [3], [7]]
)
def test_background_smoothing_box(sourcextractor, datafiles, module_output_area, smoothing_box):
    """
    Run sourcextractor with different smoothing boxesizes: from 1 (no smoothing) to more than cells are.
    Note: 1536 / 256 = 6
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid',
        detection_image=datafiles / 'sim12' / 'img' / 'sim12_r_01.fits.gz',
        background_cell_size=256,
        smoothing_box_size=smoothing_box,
        log_file='sourcextractor_{}.log'.format(smoothing_box)
    )
    assert run.exit_code == 0


@pytest.mark.regression
@pytest.mark.parametrize(
    ['smoothing_box'], [[-1]]
)
def test_background_bad_smoothing_box(sourcextractor, datafiles, module_output_area, smoothing_box):
    """
    If the smoothing box is < 0, then sourcextractor is expected to fail with a descriptive error
    """
    sourcextractor.set_output_directory(module_output_area)

    run = sourcextractor(
        output_properties='SourceIDs,PixelCentroid,WorldCentroid',
        detection_image=datafiles / 'sim12' / 'img' / 'sim12_r_01.fits.gz',
        smoothing_box_size=smoothing_box,
        log_file='sourcextractor_{}.log'.format(smoothing_box)
    )
    assert run.exit_code == 1
    assert ' in smoothing-box-size: {}'.format(smoothing_box) in run.stderr
