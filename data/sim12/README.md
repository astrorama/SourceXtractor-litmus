# SIM12
For the single frame tests, we use the image 01
from the R band from the simulation 12.

Note that we also include the configuration that was used to
run sextractor 2: `default.param`, `default.sex` and `default.conv`

## Coadd per band

```bash
swarp sim12_r_0*.fits -SUBTRACT_BACK Y -COMBINE_TYPE AVERAGE
```

## Full coadd

```bash
swarp sim12_?.fits -COMBINE_TYPE CHI-MEAN -WEIGHT_TYPE MAP_WEIGHT -RESCALE_WEIGHTS N -RESAMPLE N
```

## Generate reference catalogs

```bash
sex img/sim12_r_01.fits -CATALOG_NAME ref/sim12_r_01_reference.fits
sex img/sim12_r.fits -CATALOG_NAME ref/sim12_r_reference.fits -WEIGHT_IMAGE img/sim12_r.weight.fits -WEIGHT_TYPE MAP_WEIGHT -RESCALE_WEIGHTS N
sex img/sim12_g.fits -CATALOG_NAME ref/sim12_g_reference.fits -WEIGHT_IMAGE img/sim12_g.weight.fits -WEIGHT_TYPE MAP_WEIGHT -RESCALE_WEIGHTS N
```
