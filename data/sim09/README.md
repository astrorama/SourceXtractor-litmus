# SIM09
For the single frame tests, we use the image 01
from the R band from the simulation 09.

Note that we also include the configuration that was used to
run sextractor 2: `default.param`, `default.sex` and `default.conv`

You can reproduce the original statistics running

```bash
./bin/sex_output_stats.py --plot --world-centroid-alpha X_IMAGE --world-centroid-delta Y_IMAGE -f FLUX_ISO -f FLUX_APER:0 -f FLUX_APER:1 -f FLUX_APER:2 -f FLUX_AUTO "data/sim09/sex2_output.fits" "data/sim09/sim09_r_01.list"
```
