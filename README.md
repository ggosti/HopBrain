# HopBrain
Full brain mass model simulation using Hopfield Recurrent Neural Networks with Schaefer 2018 1000 Parcels

## Run simulations 

To run simulations run
```
python turboBrainLambda.py
```
the lines in the script from 11 to 19 allow you to change parameters.
```python
runs = 1000#40
passi = 100#200
autapse = True
randomize = True#True #False
parcelsName = 'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
```
The stady state are saved in `SRuns-autapse.csv` and in `parametersRuns-autapse.csv` are saved the labdas used in the simulations.

## Make the Structure Functions Sd

To run simulations run
```
python makeBinnedSd.py
```
the lines in the script from 36 to 43 allow you to change parameters.
```python
runs = 1000#40
passi = 100#200
autapse = True
randomize = True#True #False
parcelsName = 'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'

fitxlim = 3.5 
nBins = 100
```

Binned strucutre function mean on runs and the standard deviation are saved as files `binnedSd_mean.npy`
and `binnedSd_std.npy`
