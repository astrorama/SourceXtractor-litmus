import hashlib
import logging
import os
import pytest
import shutil
import tempfile
from pathlib import Path
from util.bin import Executable, ExecutionResult


def pytest_addoption(parser):
    """
    Register with PyTest options for this test suite
    """
    parser.addini(
        'sextractorxx',
        default='${CMAKE_PROJECT_PATH}/SExtractorxx/build.${BINARY_TAG}/bin/SExtractor',
        help='Location of the SExtractor binary (can use environment variables)',
    )
    parser.addini(
        'data_dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
        help='Location of test data files',
    )
    parser.addini(
        'sextractorxx_defaults',
        default=os.path.join(os.path.dirname(__file__), '..', 'sextractorxx.config'),
        help='Configuration file with the default configuration for SExtractor'
    )
    parser.addini(
        'sextractorxx_output_area',
        default=tempfile.gettempdir(),
        help='Where to put the results of the SExtractor runs'
    )


@pytest.fixture(scope='session')
def datafiles(request):
    """
    Fixture for the test data directory
    :return: A pathlib.Path object
    """
    path = Path(request.config.getini('data_dir'))
    assert os.path.exists(path.name)
    return path


@pytest.fixture(scope='session')
def sextractorxx_defaults(request):
    cfg = {}
    with open(request.config.getini('sextractorxx_defaults')) as cfg_fd:
        for l in cfg_fd.readlines():
            k, v = l.strip().split('=', 2)
            cfg[k] = v
    return cfg


class SExtractorxx(Executable):
    """
    Wraps Executable so the default configuration file and additional parameters
    can be passed via a configuration file.
    """

    def __init__(self, exe, area, defaults):
        super(SExtractorxx, self).__init__(exe)
        self.__area = area
        self.__defaults = defaults
        self.__output_dir = None
        self.__output_catalog = None

    def get_output_directory(self):
        return self.__output_dir

    def get_output_catalog(self):
        return self.__output_catalog

    def run_with_config(self, *args, **kwargs):
        final_args = self.__defaults.copy()
        for k, v in kwargs.items():
            final_args[k.replace('_', '-')] = str(v)
        cfg_hash = hashlib.md5(str(sorted(final_args.items())).encode('utf-8')).hexdigest()

        self.__output_dir = Path(self.__area) / f'sextractorxx_{cfg_hash}'
        if 'output-file' not in final_args:
            final_args['output-file'] = self.__output_dir / 'output.fits'
        self.__output_catalog = final_args['output-file']

        if os.path.exists(self.__output_catalog):
            return ExecutionResult(0, '', '')

        os.makedirs(self.__output_dir, exist_ok=True)
        cfg_file = self.__output_dir / 'sextractorxx.config'
        with open(cfg_file, 'w') as cfg_fd:
            for k, v in final_args.items():
                print(f'{k}={v}', file=cfg_fd)

        try:
            result = super(SExtractorxx, self).run(
                '--config-file', cfg_file,
                '--log-level', 'WARN',
                *args
            )
        except:
            os.unlink(self.__output_catalog)
        if result.exit_code != 0:
            os.unlink(self.__output_catalog)
        return result


@pytest.fixture(scope='session')
def sextractorxx(request, sextractorxx_defaults):
    """
    Fixture for the sExtractor executable
    """
    return SExtractorxx(
        os.path.expandvars(request.config.getini('sextractorxx')),
        os.path.expandvars(request.config.getini('sextractorxx_output_area')),
        defaults=sextractorxx_defaults
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
