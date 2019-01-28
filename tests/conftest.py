import logging
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from util.bin import Executable


def pytest_addoption(parser):
    """
    Register with PyTest options for this test suite
    """
    group = parser.getgroup('litmus')
    group.addoption(
        "--sextractorxx",
        action="store",
        dest="sextractorxx",
        default="${CMAKE_PROJECT_PATH}/SExtractorxx/build.${BINARY_TAG}/bin/SExtractor",
        help='Location of the SExtractor binary (can use environment variables)',
    )
    group.addoption(
        "--data-dir",
        action="store",
        dest="data_dir",
        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
        help='Location of test data files',
    )


@pytest.fixture(scope='session')
def datafiles(request):
    """
    Fixture for the test data directory
    :return: A pathlib.Path object
    """
    path = Path(request.config.getoption('data_dir'))
    assert os.path.exists(path.name)
    return path


@pytest.fixture(scope='session', autouse=True)
def sextractorxx(request):
    """
    Fixture for the sExtractor executable
    """
    return Executable(
        os.path.expandvars(request.config.getoption('sextractorxx'))
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # From
    # https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures

    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"

    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture
def output_directory(request):
    """
    Fixture for an output directory that will be cleaned up after the test
    is done. If the test fails, it will *not* clean the directory
    """
    temp_dir = tempfile.mkdtemp(prefix='sextractorxx')

    yield Path(temp_dir)
    if request.node.rep_setup.passed and request.node.rep_call.failed:
        logging.warning(f'Test failed, keeping the output directory "{temp_dir}"')
    else:
        shutil.rmtree(temp_dir)
