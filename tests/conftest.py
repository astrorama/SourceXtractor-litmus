import hashlib
import logging
import os
import re

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
            cfg[k.replace('-', '_')] = v
    return cfg


class SExtractorxx(object):
    """
    Wraps Executable so the default configuration file and additional parameters
    can be passed via a configuration file.
    """

    def __init__(self, exe, output_dir, defaults):
        self.__exe = exe
        self.__output_dir = output_dir
        self.__defaults = defaults
        self.__output_catalog = None

    def get_output_directory(self):
        return self.__output_dir

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
                if v is not None:
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
        if self.__output_catalog and os.path.exists(self.__output_catalog):
            return ExecutionResult(0, '', '')

        result = self.__exe.run(
            '--log-level', 'WARN',
            *cmd_args
        )

        if result.exit_code != 0 and self.__output_catalog and os.path.exists(self.__output_catalog):
            os.unlink(self.__output_catalog)

        return result


@pytest.fixture
def sextractorxx(request, sextractorxx_defaults):
    """
    Fixture for the sExtractor executable
    """
    exe = Executable(os.path.expandvars(request.config.getini('sextractorxx')))
    area = Path(os.path.expandvars(request.config.getini('sextractorxx_output_area')))
    if request.fixturenames[0] != request.fixturename:
        output_dir = area / re.sub('[\[\]]', '_', request.fixturenames[0])
    else:
        output_dir = area / re.sub('[\[\]]', '_', request.node.name)
    return SExtractorxx(exe, output_dir, sextractorxx_defaults)
