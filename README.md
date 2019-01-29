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

## Parametrized tests
Some tests are parametrized based on a column, error, etc.
This avoids writing two separate tests that do basically
the same thing but based on different parameters.

For instance, when testing the photometry, we may be using
the isophotal flux, the aperture, the auto aperture, etc.
But, ultimately, they are all compared in the same manner
with the original 'stuff' simulation.

For these cases, you will see something like

```python
@pytest.mark.parametrize(
    ['flux_column', 'sum_squared_errors'], [
        ['auto_flux', 16400],
        ['isophotal_flux', 23000],
    ]
)
```
 
That means the test is going to run twice, once for 
`auto_flux`, checking the sum of the squared errors is less
than 16400, and another for `isophotal_flux`, where the error
limit is 23000.

## Where do these acceptable errors come from, anyway?
From equivalent runs from SExtractor 2. SExtractor++ should
do *at least as good*, so these tests try to make sure
not only that it runs, but that the emitted values are good enough. 
