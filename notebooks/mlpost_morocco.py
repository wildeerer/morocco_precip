#PASS IN A TRUE AND PREDICTED 
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class precip_postprocessing:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.location_dict = None

    def load_data(self, observed_fp, predicted_fp, verbose = False): #Loading from the file path
        self.observed_data = xr.open_dataset(observed_fp)
        self.predicted_data = xr.open_dataset(predicted_fp)
        if verbose:
            print(f'CHIRPS NetCDF File: \n{self.observed_data}')
            print(f'NWP NetCDF File: \n{self.predicted_data}')

    def load_model(self, model):
        self.model = model

    def load_scaler(self, scaler):
        self.scaler = scaler

    def preprocess(self, feature_array): #Must require load_data first
        self.observed = self.observed_data.to_dataframe().reset_index()
        self.predicted = self.predicted_data.to_dataframe().reset_index()

        self.merged = pd.merge(self.observed, self.predicted, on=['Time', 'Lat', 'Lon'], how = 'left').dropna() # <---- THIS IS SPECIFIC TO THIS USE CASE ---- not universal bc of feature names
        


    def add_month_column(self): #Must require preprocess done first.. find 
        df = self.merged
        df['Month'] = df['Time']%100

    def merge_location_dict(self, d):
        self.location_dict = pd.DataFrame(d).T.reset_index()
        self.merged_df = 0


    # def train_split(self, use_loc_dict = False):
    #     if use_loc_dict == True:
    #         self.X = self.

    #     else:

    

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split()

    def predict(self):
        return 0
    
    def evaluate(self):
        return 0