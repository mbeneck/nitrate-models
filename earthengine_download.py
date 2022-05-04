# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:33:56 2022

@author: eckmb
"""

import ee
import pandas as pd

def get_neracoos_latlon(neracoos_dataframe):
    return neracoos_dataframe['latitude'][0], neracoos_dataframe['longitude'][0]

def read_neracoos_data(path):
    df = pd.read_csv(path, skiprows=[1])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index(['time'], inplace=True)
    df.index = df.index.tz_convert(None) 
    return df

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df

def get_ee_timeseries(imagecollection, bandlist, startdate, enddate, lat, lon):
    collection = ee.ImageCollection(imagecollection).filterDate(startdate, enddate)
    pt = ee.Geometry.Point(lon, lat)
    data = collection.getRegion(pt, 1000).getInfo()
    return ee_array_to_df(data, bandlist)

def get_ee_timeseries_for_neracoos(neracoos_dataframe, imagecollection, bandlist, startdate, enddate):
    latlon = get_neracoos_latlon(neracoos_dataframe)
    return get_ee_timeseries(imagecollection, bandlist, startdate, enddate, latlon[0], latlon[1])

def clean_ee_timeseries(ee_timeseries):
    ee_timeseries['datetime'] = pd.to_datetime(ee_timeseries['datetime'])
    ee_timeseries.set_index('datetime', inplace=True)
    return ee_timeseries.resample('1d').mean().rolling('4d', min_periods=1).mean().dropna()     # computes a 4d rolling average to match timescale of NERACOOS, SMAP data
    
start = '2011-01-01'
end = '2022-05-02'
MODIS_L3 = 'NASA/OCEANDATA/MODIS-Aqua/L3SMI'
MODIS_L3_bands = ['chlor_a', 'nflh', 'poc', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645', 'Rrs_667', 'Rrs_678', 'sst']

E1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\E01_corrected_nitrate_csv_ea04_b27f_db2e.csv')
GB = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\GREAT_BAY_TS_corrected_nitrate_csv_fe8c_cf13_11fe.csv')
I1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\I01_corrected_nitrate_csv_da15_4f16_7fa2.csv')
M1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\M01_corrected_nitrate_csv_da15_4f16_7fa2.csv')
N1 = read_neracoos_data('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\NERACOOS\\N01_corrected_nitrate_csv_da15_4f16_7fa2.csv')

ee.Initialize()     # initialization code for Earth Engine python API

# modis = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI').filterDate('2011-01-01', '2022-05-02')
# E1_pt = ee.Geometry.Point(get_neracoos_latlon(E1)[1], get_neracoos_latlon(E1)[0])
# modis_E1 = modis.getRegion(E1_pt, 1000).getInfo()
# modis_E1_df = ee_array_to_df(modis_E1, ['sst'])

E1_modis = clean_ee_timeseries(get_ee_timeseries_for_neracoos(E1, MODIS_L3, MODIS_L3_bands, start, end))
GB_modis = clean_ee_timeseries(get_ee_timeseries_for_neracoos(GB, MODIS_L3, MODIS_L3_bands, start, end))
I1_modis = clean_ee_timeseries(get_ee_timeseries_for_neracoos(I1, MODIS_L3, MODIS_L3_bands, start, end))
M1_modis = clean_ee_timeseries(get_ee_timeseries_for_neracoos(M1, MODIS_L3, MODIS_L3_bands, start, end))
N1_modis = clean_ee_timeseries(get_ee_timeseries_for_neracoos(N1, MODIS_L3, MODIS_L3_bands, start, end))

E1_modis.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\E1.csv')
GB_modis.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\GB.csv')
I1_modis.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\I1.csv')
M1_modis.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\M1.csv')
N1_modis.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\MODIS\\N1.csv')
