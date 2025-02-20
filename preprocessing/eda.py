# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os

CHIRPS_FP='../data/raw/CHIRPS_1991_2021_MOR.nc'
RAW_FP='../data/raw/RAW_CFSv2_1991_2020_MOR.nc'

chirps_raw=xr.open_dataset(CHIRPS_FP)
model_raw=xr.open_dataset(RAW_FP)

tp_chirps_at_time = chirps_raw.isel(Time=0)
tp_raw_at_time=model_raw.isel(Time=0)

xr.plot.scatter(model_raw,"Time","TP")

tp_chirps_at_time['TP'].plot()
#plt.show()
