# HopBrain
Full brain mass model simulation using Hopfield Recurrent Neural Networks with Schaefer 2018 1000 Parcels

## Dowload the parcellations

To dowload the parcellations run the python script
```
python getParcellations.py
```
the script may have problems in Windows, in this case, download the following csv files:
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_500Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_600Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_700Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_800Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
- https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_900Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv


## Run simulations 

To run simulations run
```
python turboBrainLambda.py
```
the lines in the script from 11 to 19 allow you to change parameters. Change N from 40 to 1000 to reproduce the papers results.
```python
runs = 40 #1000#40
passi = 100#200
autapse = True
randomize = True#True #False
parcelsName = 'Centroid_coordinates/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
```
The stady state are saved in `SRuns-autapse.csv` and in `lamdaValues-autapse.csv` are saved the labdas used in the simulations.

## Make the Structure Functions Sd

To run simulations run
```
python makeBinnedSd.py
```
the lines in the script from 36 to 43 allow you to change parameters. Change N from 40 to 1000 to reproduce the papers results.
```python
runs = 40 #1000#40
passi = 100 #200
autapse = True
randomize = True#True #False
parcelsName = 'Centroid_coordinates/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'

fitxlim = 3.5 
nBins = 100
```

Binned strucutre function mean on runs and the standard deviation are saved as files `binnedSd_mean.npy`
and `binnedSd_std.npy`

## Run simulations cutting edges 

To run simulations run cutting edges somaller than certain thresholds
```
python turboBrainLambda-ThJ.py
```
the lines in the script from 11 to 19 allow you to change parameters. Change N from 40 to 1000 to reproduce the papers results.
```python
runs = 40 #1000#40
passi = 100#200
autapse = True
randomize = True#True #False
parcelsName = 'Centroid_coordinates/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
```
Line 54 is were thresholds are set 
```python
ths = [0.0001,0.0002,0.0004,0.001,0.002,0.004,0.01,0.02,0.04,.1,.2,.4,.8,.9]
```
The stady state are saved in `SRuns-autapse-thJ-x.xx.csv` and in `lamdaValues-autapse-thJ.csv` are saved the labdas used in the simulations.

## Make the Structure Functions Sd for simulations with cutted edges

```
python makeBinnedSd-ThJ.py
```
the lines in the script from 36 to 43 allow you to change parameters. Change N from 40 to 1000 to reproduce the papers results.
```python
runs = 40 #1000#40
passi = 100#200
autapse = True
randomize = True#True #False
parcelsName = 'Centroid_coordinates/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'

fitxlim = 3.5 
nBins = 100
```
Line 54 is were thresholds are set, thresholds should be the same as in the simulations run,
```python
ths = [0.0001,0.0002,0.0004,0.001,0.002,0.004,0.01,0.02,0.04,.1,.2,.4,.8,.9]
```
Binned strucutre function mean on runs and the standard deviation are saved as files `binnedSd_mean.npy`
and `binnedSd_std.npy`

## Make plots for degree analysis

For producing degree analysis plots run:
```
python connectivityLambda.py
```
the lines in the script from 16 to 18 allow you to change parameters.
```
autapse = True
randomize = False #True #False #True 
parcelsName = 'Centroid_coordinates/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
```


## Generate plots

The jupyter notebook `Plot3D.ipynb` allows to generate the 3D paercel models.
The jupyter notebook `PlotData3.ipynb` allows to generate the analysis plots.

## Citation

If you find this project useful, please cite:

> Giorgio Gosti, Sauro Succi, Giancarlo Ruocco. 
> *Evidence of Scaling Regimes in the Hopfield Dynamics of Whole Brain Model*,
> arXiv, 2024, https://arxiv.org/abs/2401.07538.
>

## DOI

[![DOI](https://zenodo.org/badge/611266001.svg)](https://zenodo.org/doi/10.5281/zenodo.12799027)
