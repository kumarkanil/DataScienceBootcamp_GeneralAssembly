#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 18: Databases and SQL
'''

import os

import numpy as np
import pandas as pd

import sqlite3

CYCLE_HIRES_URL = 'https://files.datapress.com/london/dataset/number-bicycle-hires/2016-05-17T09:24:57/tfl-daily-cycle-hires.xls'

# Load TfL Cycle Hire dataset
hires = pd.read_excel(CYCLE_HIRES_URL, sheetname='Data')

# Convert 'Day' to 'datetime' and set as index
hires.Day = pd.to_datetime(hires.Day, unit='D')
hires.set_index('Day', inplace=True)

# Extract first column (daily hires) and convert to 'float'
hires = hires.iloc[:,0].astype('float').to_frame('Hires')

# Write DataFrame to disk
db_connection = sqlite3.connect(os.path.join('datasets', 'tfl_cycle_hire.db'))
hires.to_sql('hires', con=db_connection, if_exists='replace')

# Retrieve first 10 records
pd.io.sql.read_sql('SELECT * FROM hires LIMIT 10', con=db_connection)

# Retrieve all records from 2016
pd.io.sql.read_sql(
    '''SELECT Hires
       FROM hires
       WHERE STRFTIME(\'%Y\', Day) == \'2016\'''', con=db_connection)

# Compute average number of cycle hires by month
pd.io.sql.read_sql(
    '''SELECT Day, AVG(Hires) AS average_hires
       FROM hires
       GROUP BY STRFTIME(\'%m\', Day)''', con=db_connection)

# Aggregate records by year and month, then sort by total number of cycle hires
pd.io.sql.read_sql(
    '''SELECT Day, SUM(Hires) AS total_hires
       FROM hires
       GROUP BY STRFTIME(\'%Y%m\', Day)
       ORDER BY total_hires DESC''', con=db_connection)

