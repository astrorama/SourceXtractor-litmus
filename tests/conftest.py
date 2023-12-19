import logging
import os
import shutil
import sys
from configparser import ConfigParser
from pathlib import Path

import pytest

from util.bin import Executable

logger = logging.getLogger(__name__)


@pytest.fixture(scope='session')
def test_configuration(request):
    config = ConfigParser()
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
    area = Path(os.path.expandvars(test_configuration.get('sourcextractor', 'output_area')))
    for c in request.node.listchain():
        if isinstance(c, pytest.Module):
            area = area / os.path.basename(c.name)
    return area


@pytest.fixture(scope='session')
def sourcextractor_defaults(test_configuration, datafiles):
    os.environ['DATADIR'] = str(datafiles)
    cfg = {}
    with open(test_configuration.get('sourcextractor', 'defaults')) as cfg_fd:
        for l in cfg_fd.readlines():
            if '#' in l:
                l, _ = l.split('#', 2)
            try:
                k, v = l.strip().split('=', 2)
                cfg[k.replace('-', '_')] = os.path.expandvars(v)
            except ValueError:
                pass
    return cfg


class SourceXtractor(object):
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
        self.__env['OMP_DYNAMIC'] = 'false'
        self.__env['OMP_NUM_THREADS'] = '1'

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
            if 'output_catalog_filename' not in cfg_args:
                cfg_args['output_catalog_filename'] = self.__output_dir / 'output.fits'
            cfg_file = self.__output_dir / 'sourcextractor.config'
            with open(cfg_file, 'w') as cfg_fd:
                for k, v in cfg_args.items():
                    k = k.replace("_", "-")
                    if isinstance(v, list):
                        for sv in v:
                            print(f'{k}={sv}', file=cfg_fd)
                    elif v is not None:
                        print(f'{k}={v}', file=cfg_fd)
            cmd_args.extend(['--config-file', cfg_file])

        self.__output_catalog = cfg_args.get('output_catalog_filename', None)
        if self.__output_catalog and os.path.exists(self.__output_catalog):
            logger.debug(f'Overwriting {self.__output_catalog}')
        result = self.__exe.run(*cmd_args, cwd=self.__output_dir, env=self.__env)

        if result.exit_code != 0 and self.__output_catalog and os.path.exists(self.__output_catalog):
            os.unlink(self.__output_catalog)

        return result


@pytest.fixture(scope='module')
def sourcextractor(request, test_configuration, sourcextractor_defaults, module_output_area):
    """
    Fixture for the SourceXtractor executable
    """
    expanded = os.path.expandvars(test_configuration.get('sourcextractor', 'binary'))
    which = shutil.which(expanded)
    if which is None:
        raise RuntimeError(f'Could not find {expanded}')
    exe = Executable(shutil.which(os.path.expandvars(test_configuration.get('sourcextractor', 'binary'))))

    test_output_area = module_output_area / request.node.name
    return SourceXtractor(
        exe, os.path.expandvars(test_configuration.get('sourcextractor', 'pythonpath')),
        test_output_area,
        sourcextractor_defaults
    )


@pytest.fixture(scope='session')
def sourcextractor_py(test_configuration):
    python_path = os.path.expandvars(test_configuration.get('sourcextractor', 'pythonpath')).split(':')
    for pp in python_path:
        if pp not in sys.path:
            sys.path.append(pp)
    mod = __import__('sourcextractor.config')
    return mod.config


@pytest.fixture(scope='session')
def tolerances(test_configuration):
    """
    Allow to configure the tolerances for the checks
    """
    return {
        'magnitude': float(test_configuration.get('tolerances', 'magnitude')),
        'distance': float(test_configuration.get('tolerances', 'distance')),
        'signal_to_noise': float(test_configuration.get('tolerances', 'signal_to_noise'))
    }


@pytest.fixture(scope='session')
def simulation_mag_zeropoint(test_configuration):
    return float(test_configuration.get('simulation', 'mag_zeropoint'))


@pytest.fixture(scope='session')
def simulation_exposure(test_configuration):
    return float(test_configuration.get('simulation', 'exposure'))
