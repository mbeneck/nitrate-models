# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:23:48 2022

@author: eckmb
"""

import pandas as pd
import ee

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    # df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df.rename({'datetime':'modis_datetime', 'time':'modis_time'}, inplace=True, axis=1)

    # Keep the columns of interest.
    df = df[['modis_time','modis_datetime',  *list_of_bands]]

    return df

def get_ee_point_from_GOM_idx(idx, imagecollection, bandlist):
    date = idx[0]
    lat = idx[1]
    lon = idx[2]
    nextdate = date + pd.Timedelta('1d')
    return get_ee_timeseries(imagecollection, bandlist, date.strftime('%Y-%m-%d'), nextdate.strftime('%Y-%m-%d'), lat, lon).loc[0]

def get_ee_df_for_GOM(GOM_df, imagecollection, bandlist):
    GOM_index = GOM_df.index
    result = pd.DataFrame(index = GOM_index, columns=bandlist)
    
    missedrows=0
    for i, idx in enumerate(GOM_index):
        print('Getting image %i of %i' % (i, GOM_index.size-1))
        try:
            result.loc[idx] = get_ee_point_from_GOM_idx(idx, imagecollection, bandlist)
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # __str__ allows args to be printed directly,
                                 # but may be overridden in exception subclasses
            missedrows += 1
            print('Oops! Missed row %i of %i. Total missed rows: %i' %(i, GOM_index.size-1, missedrows))
    return result
    
def get_ee_timeseries(imagecollection, bandlist, startdate, enddate, lat, lon):
    collection = ee.ImageCollection(imagecollection).filterDate(startdate, enddate)
    pt = ee.Geometry.Point(lon, lat)
    data = collection.getRegion(pt, 1000).getInfo()
    return ee_array_to_df(data, bandlist)


def clean_ee_timeseries(ee_timeseries):
    ee_timeseries['datetime'] = pd.to_datetime(ee_timeseries['datetime'])
    ee_timeseries.set_index('datetime', inplace=True)
    return ee_timeseries.resample('1d').mean().rolling('4d', min_periods=1).mean().dropna()

ee.Initialize()

MODIS_L3 = 'NASA/OCEANDATA/MODIS-Aqua/L3SMI'
MODIS_L3_bands = ['chlor_a', 'nflh', 'poc', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645', 'Rrs_667', 'Rrs_678', 'sst']


GOM = pd.read_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Rebuck\\RebuckGoMaineNutrients.txt', names=['month', 'day', 'year', 'lon', 'lat', 'bottom depth', 'sample depth', 'temp', 'salinity', 'Nitrate', 'Silicate', 'Phosphate', 'Chlorophyll', 'Phosphate Quality', 'Silicate Quality', 'Nitrate Quality'], index_col=False, sep='\t')
GOM = GOM.apply(pd.to_numeric, errors='coerce')
GOM['date'] = pd.to_datetime(GOM[['year', 'month', 'day']])
GOM.set_index('date', inplace=True)

GOM_shallow = GOM.loc[(GOM['sample depth'] <= 5) & (GOM['Nitrate Quality'] == 0) & (GOM['salinity'].isna() == False) & (GOM['temp'].isna() == False)]
GOM_shallow_ave = GOM_shallow.groupby(['date', 'lat', 'lon']).mean()
GOM_shallow_ave.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\gom_shallow_ave.csv')


modis = get_ee_df_for_GOM(GOM_shallow_ave['2002-07-03':'2022-05-03'], MODIS_L3, MODIS_L3_bands).dropna(how='all')
GOM_shallow_ave_modis = GOM_shallow_ave.join(modis, how='inner')
GOM_shallow_ave_modis.to_csv('C:\\Users\\eckmb\\OneDrive - Northeastern University\\Courses\\CIVE5280\\Final Project\\Processed Data\\gom_shallow_ave_modis.csv')


