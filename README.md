# SExtractor++ test suite

* `tests` contains the tests for SExtractor. Just execute `pytest`
  (Python 3 version!) on the project root directory to run them.  
* `bin` contains a set of utilities required to prepare the tests.
  For instance, to gather some statistics from a sextractor run that can
  be used to verify that SExtractor++ is, at least, as good.

## Markers
Tests are marked so different subsets can be executed separately.
For instance, tests that take a long time to run SExtractor are
marked as `pytest.mark.slow`, so it would be a bad idea to run them on
each commit, for instance. They can be filtered out like

```bash
py.test-3 -m "not slow"
```

Markers used in the test suite:

* `pytest.mark.slow`
* `pytest.mark.regression`
