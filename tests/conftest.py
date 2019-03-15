import os
import sys
from pathlib import Path

import pytest
from iniparse import SafeConfigParser

from util import stuff
from util.bin import Executable


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
def sextractorxx_py(test_configuration):
    python_path = os.path.expandvars(test_configuration.get('sextractorxx', 'pythonpath')).split(':')
    for pp in python_path:
        if pp not in sys.path:
            sys.path.append(pp)
    mod = __import__('sextractorxx.config')
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
