#!/usr/bin/env python

'''
GA Data Science Q2 2016

In-class exercise 8: SQL
'''

import os

import numpy as np
import pandas as pd

import sqlite3

# Open connection to database
# (Download from http://chinookdatabase.codeplex.com/)
db_connection = sqlite3.connect(os.path.join('datasets', 'Chinook_Sqlite.sqlite'))

# Select the first 10 records from 'customer'

# Select the first name of all customers from the UK

# Select the city and country of all customers from the UK or Portugal

# Select the first 10 records from 'invoice'

# Join 'customer' and 'invoice', and retrieve customer ID and invoice amount

# (Continued) Compute the total of all invoices by customer

# (Continued) Aggregate only invoices from 2013

# (Continued) Order by total amount in descending order

# (Continued) Add the first name of the support rep from table 'employee'

