!pip install numpy
!pip install pandas
!pip install statsmodels

import pandas as pd # data manipulation library
import numpy as np # math library
import datetime as dt # to discover week day
import statsmodels as sm # statistical models
import statsmodels.api as sma # statistical models api
import time as tm
import matplotlib.pyplot as plt


### Open dataset

DATASET_PATH = f"dataset/all_data_sorted.csv"
col_names = [
  'Sensor',
  'Date',
  'Time',
  'Lane',
  'Speed',
  'Max Speed',
  'Size'
]

alldata = pd.read_csv(DATASET_PATH, ';', header=None, names=col_names)

# Get specific sensor
data = alldata[alldata['Sensor'] == 'RSI128']

# Remove unnecessary columns
data = data.drop(columns=['Sensor','Lane','Max Speed','Size'])

### Create week day column from date

# Get datetime
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')

# Adjust type
f = lambda x : tm.strptime(x, '%H:%M:%S')
data['Time'] = data['Time'].apply(f)
g = lambda x : dt.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
data['Time'] = data['Time'].apply(g)
h = lambda x : int(x)
data['Time'] = data['Time'].apply(h)

# Create week day from date
j = lambda x : x.weekday()
data['WeekDay'] = data['Date'].apply(j)

### Save to file
data.to_csv(f"data/dataset.csv", ";", index=False)

### Get flow
flows_intervals = [150, 300, 450, 900]

for flow_interval in flows_intervals:
  flow_data = get_flow(data, flow_interval)
  week_size = get_week_size(flow_interval)

  flow_data.to_csv(f"dataset/dataset_flow_{flow_interval}.csv", ";", index=False)

### Plot flows
def plot_flow(flow_series, flow_interval):
  """ Plot of Flow
  
  Plot the flow from week to week
  
  Arguments:
    flow_series: an array of flows
    flow_interval: the interval in which the flow was made
  """

  week_size = get_week_size(flow_interval)
  n = len(flow_series) // week_size

  if len(flow_series) % week_size == 0:
    print('Yey')

  for i in range(n):
    s = week_size * i
    e = min(s + week_size, len(flow_series))
    path = f"plots/flow/flow_{flow_interval}_week_{str(i+1).zfill(2)}"

    plt.plot(flow_series[s:e])

    plt.title(f"Fluxo (Intervalo de {flow_interval} segundos) - Semana {i+1}")
    plt.ylabel('Fluxo')
    plt.xlabel('Tempo')
    plt.rcdefaults()
    
    plt.savefig(path + ".png", bbox_inches='tight')
    # plt.savefig(path + ".pdf")
    
    plt.close('all')

flows_intervals = [150, 300, 450, 900]

for flow_interval in flows_intervals:
  flow_data = get_flow(data, flow_interval)
  plot_flow(flow_data['Flow'], flow_interval)