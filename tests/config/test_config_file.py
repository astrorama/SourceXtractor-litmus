from tempfile import NamedTemporaryFile


def test_missing_config_file(sourcextractor):
    """
    Pass a non-existing configuration file.
    """
    run = sourcextractor(config_file='/etc/this/does/not/exist.config')
    assert run.exit_code > 0
    assert 'doesn\'t exist' in run.stderr


def test_malformed_config_file(sourcextractor):
    """
    Pass a malformed configuration file
    """
    with NamedTemporaryFile() as config:
        config.write(bytes('a:b', encoding='ascii'))
        config.flush()
        run = sourcextractor(config_file=config.name)
    assert run.exit_code > 0
    assert 'invalid line' in run.stderr
