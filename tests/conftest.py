import os
import re

import pytest
import tempfile
from pathlib import Path

from iniparse import SafeConfigParser

from util.bin import Executable, ExecutionResult
from util import stuff

if False:
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
        parser.addini(
            'sextractorxx_mag_zeropoint',
            default=26.,
            help='Magnitude zeropoint used to convert fluxes to magnitudes'
        )
        parser.addini(
            'sextractorxx_exposure',
            default=300.,
            help='Exposure time used to convert fluxes to magnitudes'
        )
        parser.addini(
            'sextractorxx_signal_to_noise',
            default=1000.,
            help='Signal to noise ratio used to filter the catalogs'
        )


@pytest.fixture(scope='session')
def test_configuration(request):
    config = SafeConfigParser()
    config.read([request.config.inifile])
    return config


@pytest.fixture(scope='session')
def datafiles():
    """
    Fixture for the test data directory
    :return: A pathlib.Path object
    """
    path = Path(os.path.join(os.path.dirname(__file__), '..', 'data'))
    assert os.path.exists(path.name)
    return path


@pytest.fixture(scope='module')
def module_output_area(request, test_configuration):
    """
    Generate a path where a test module should store its output files
    """
    area = Path(os.path.expandvars(test_configuration.get('sextractorxx', 'output_area')))
    for c in request.node.listchain():
        if isinstance(c, pytest.Module):
            area = area / c.name
    return area


@pytest.fixture(scope='session')
def sextractorxx_defaults(test_configuration, datafiles):
    os.environ['DATADIR'] = str(datafiles)
    cfg = {}
    with open(test_configuration.get('sextractorxx', 'defaults')) as cfg_fd:
        for l in cfg_fd.readlines():
            if '#' in l:
                l, _ = l.split('#', 2)
            try:
                k, v = l.strip().split('=', 2)
                cfg[k.replace('-', '_')] = os.path.expandvars(v)
            except ValueError:
                pass
    return cfg


class SExtractorxx(object):
    """
    Wraps Executable so the default configuration file and additional parameters
    can be passed via a configuration file.
    """

    def __init__(self, exe, pythonpath, output_dir, defaults):
        self.__exe = exe
        self.__output_dir = output_dir
        self.__defaults = defaults
        self.__output_catalog = None
        self.__env = os.environ.copy()
        self.__env['PYTHONPATH'] = self.__env.get('PYTHONPATH', '') + ':' + pythonpath

    def get_output_directory(self):
        return self.__output_dir

    def set_output_directory(self, output_dir):
        self.__output_dir = output_dir

    def get_output_catalog(self):
        return self.__output_catalog

    def __call__(self, *args, **kwargs):
        os.makedirs(self.__output_dir, exist_ok=True)

        cmd_args = list(args)
        cfg_args = self.__defaults.copy()
        cfg_args.update(kwargs)

        # Config file provided by the test
        # Anything extra pass via command line
        if 'config_file' in cfg_args:
            cmd_args.extend(['--config-file', cfg_args.pop('config_file')])
            for k, v in cfg_args.items():
                if v.lower() == 'true':
                    cmd_args.extend([f'--{k.replace("_", "-")}'])
                elif v is not None and v.lower() != 'false':
                    cmd_args.extend([f'--{k.replace("_", "-")}', v])
        # Generate a config file with all settings
        else:
            if 'output_file' not in cfg_args:
                cfg_args['output_file'] = self.__output_dir / 'output.fits'
            cfg_file = self.__output_dir / 'sextractorxx.config'
            with open(cfg_file, 'w') as cfg_fd:
                for k, v in cfg_args.items():
                    if v is not None:
                        print(f'{k.replace("_", "-")}={v}', file=cfg_fd)
            cmd_args.extend(['--config-file', cfg_file])

        self.__output_catalog = cfg_args.get('output_file', None)
        result = self.__exe.run(*cmd_args, cwd=self.__output_dir, env=self.__env)

        if result.exit_code != 0 and self.__output_catalog and os.path.exists(self.__output_catalog):
            os.unlink(self.__output_catalog)

        return result


@pytest.fixture
def sextractorxx(request, test_configuration, sextractorxx_defaults, module_output_area):
    """
    Fixture for the SExtractor executable
    """
    exe = Executable(os.path.expandvars(test_configuration.get('sextractorxx', 'binary')))

    test_output_area = module_output_area / request.node.name
    return SExtractorxx(
        exe, os.path.expandvars(test_configuration.get('sextractorxx', 'pythonpath')),
        test_output_area,
        sextractorxx_defaults
    )


@pytest.fixture(scope='session')
def flux2mag(test_configuration):
    """
    Fixture to convert fluxes to magnitudes using the values configured in the ini file
    """
    zeropoint = float(test_configuration.get('flux2mag', 'mag_zeropoint'))
    exposure = float(test_configuration.get('flux2mag', 'exposure'))

    def _flux2mag_wrapper(v):
        return stuff.flux2mag(v, zeropoint, exposure)

    return _flux2mag_wrapper


@pytest.fixture(scope='session')
def signal_to_noise_ratio(test_configuration):
    """
    This is the signal to noise ratio used to filter out faint sources
    """
    return float(test_configuration.get('filters', 'signal_to_noise'))


@pytest.fixture(scope='session')
def tolerances(test_configuration):
    """
    Allow to configure the tolerances for the checks
    """
    return {
        'magnitude': float(test_configuration.get('tolerances', 'magnitude')),
        'flux': float(test_configuration.get('tolerances', 'flux')),
        'flux_error': float(test_configuration.get('tolerances', 'flux_error'))
    }
