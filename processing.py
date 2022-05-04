# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:48:35 2022

@author: eckmb
"""

from netCDF4 import Dataset, num2date
import pandas as pd
import ee

def get_neracoos_latlon(neracoos_dataframe):
    return neracoos_dataframe['latitude'][0], neracoos_dataframe['longitude'][0]

def nearest_idx(array, value):
    idx=(abs(array-value)).argmin()
    return idx

def extract_nc_point_timeseries(latlon, nc, varname):
    lats = nc.variables['latitude'][:]
    lons = nc.variables['longitude'][:]
    iy = nearest_idx(lats, latlon[0])
    ix = nearest_idx(lons, latlon[1])
    
    dates = num2date(nc['time_agg'], units=nc['time_agg'].units, calendar=nc['time_agg'].calendar, only_use_python_datetimes=True, only_use_cftime_datetimes=False)
    
    data = pd.DataFrame(index = pd.to_datetime(dates), columns=[varname])
    
    data[varname]=nc.variables[varname][:, iy, ix]
    
    return data.dropna()

def read_neracoos_data(path):
    df = pd.read_csv(path, skiprows=[1])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index(['time'], inplace=True)
    df.index = df.index.tz_convert(None) 
    return df

# performs a centered 4d resample of neracoos data to smap data
def resample_neracoos_to_smap(smap_df, neracoos_df):
    start_date = smap_df.index[0]
    resampled = neracoos_df.groupby('depth').resample('4d', kind='timestamp', origin=start_date, offset=pd.Timedelta(2, 'days')).mean().reset_index(level='depth', drop=True).dropna()
    resampled.index = resampled.index + pd.Timedelta(2, 'days')
    return resampled.dropna()

# joins all resulting dataframes for one master dataframe
def join_modis_smap_neracoos(neracoos, smap, modis):
    sss= extract_nc_point_timeseries(get_neracoos_latlon(neracoos), smap, 'sss')
    neracoos_resampled = resample_neracoos_to_smap(sss, neracoos)
    merged = neracoos_resampled.join(sss, how='inner')
    merged.drop(labels=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'],axis=1, inplace=True)
    return merged.join(modis, how='inner')
    


E1_modis = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\E1.csv', index_col='datetime', parse_dates=True)
GB_modis = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\GB.csv', index_col='datetime', parse_dates=True)
I1_modis = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\I1.csv', index_col='datetime', parse_dates=True)
M1_modis = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\M1.csv', index_col='datetime', parse_dates=True)
N1_modis = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\N1.csv', index_col='datetime', parse_dates=True)

E1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\E01_corrected_nitrate_csv_ea04_b27f_db2e.csv')
GB = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\GREAT_BAY_TS_corrected_nitrate_csv_fe8c_cf13_11fe.csv')
I1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\I01_corrected_nitrate_csv_da15_4f16_7fa2.csv')
M1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\M01_corrected_nitrate_csv_da15_4f16_7fa2.csv')
N1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\N01_corrected_nitrate_csv_da15_4f16_7fa2.csv')

smap = Dataset('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\SMAP\\SalinityDensity_OISSS_L4_multimission_7day_v1.nc', 'r', Format='NETCDF4')

E1_merged = join_modis_smap_neracoos(E1, smap, E1_modis)
M1_merged = join_modis_smap_neracoos(M1, smap, M1_modis)
N1_merged = join_modis_smap_neracoos(N1, smap, N1_modis)

# These were attempted but abandoned because SMAP data were not available for the buoy location.
# GB_merged = join_modis_smap_neracoos(GB, smap, GB_modis)
# I1_merged = join_modis_smap_neracoos(I1, smap, I1_modis)

E1_merged.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\E1_combined.csv')
M1_merged.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\M1_combined.csv')
N1_merged.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\N1_combined.csv')