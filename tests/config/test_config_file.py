from tempfile import NamedTemporaryFile


def test_missing_config_file(sextractorxx):
    """
    Pass a non-existing configuration file.
    """
    run = sextractorxx(config_file='/etc/this/does/not/exist.config')
    assert run.exit_code > 0
    assert 'does not exist' in run.stderr


def test_malformed_config_file(sextractorxx):
    """
    Pass a malformed configuration file
    """
    with NamedTemporaryFile() as config:
        config.write(bytes('a:b', encoding='ascii'))
        config.flush()
        run = sextractorxx(config_file=config.name)
    assert run.exit_code > 0
    assert 'invalid line' in run.stderr
