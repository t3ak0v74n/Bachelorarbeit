from tmApiClient import TmArchiveGet
import pandas as pd
from datetime import datetime, timezone, timedelta
from astropy import units as u
from astropy import coordinates as coord
from astropy.time import Time
import numpy as np
import json
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# time interval
end_date = datetime(2022, 8, 25, 12, 37, tzinfo=timezone.utc)  # datetime.utcnow()-timedelta(days=0.5)#
start_date = end_date - timedelta(days=20)

'''# Load data (deserialize)
with open('gps_measurements.pickle', 'rb') as handle:
    gps_measurements = pickle.load(handle)'''

with open("UserDataIRS.json" , "r") as UD_IRS:
    userDataIRS = json.load(UD_IRS)

userIRS= userDataIRS['user']
password=userDataIRS['pw']

# get archive, fetch data
tm_archive = TmArchiveGet(ip="https://tm.buggy.irs.uni-stuttgart.de", user= userIRS, pw= password)
tm_ses = tm_archive.createSession()

print("fetching between %s and %s" % (start_date, end_date))
"""gps_measurements = tm_archive.getTmParametersAsPandas(begin=start_date, end=end_date,
                                                      Parameters=["AYTPOS00", "AYTPOS01", "AYTPOS02", "AYTVEL00",
                                                                  "AYTVEL01", "AYTVEL02"],
                                                      live=False, session=tm_ses)"""
gps_measurements=tm_archive.getTmParametersAsPandas(start_date, end_date,["AYTPOS00", "AYTPOS01", "AYTPOS02", "AYTVEL00",
                                                                  "AYTVEL01", "AYTVEL02"], False, tm_ses )




if gps_measurements['timestamp'].dtype == 'int64':
    gps_measurements['timestamp'] = pd.to_datetime(gps_measurements['timestamp'], unit='ms', utc=True)

#every 20 measurments
#gps_measurements = gps_measurements.iloc[::20]

# DataFrames with t, p, v
df = pd.DataFrame({
    'time': gps_measurements['timestamp'],
    'position_x': gps_measurements['AYTPOS00'],
    'position_y': gps_measurements['AYTPOS01'],
    'position_z': gps_measurements['AYTPOS02'],
    'velocity_x': gps_measurements['AYTVEL00'],
    'velocity_y': gps_measurements['AYTVEL01'],
    'velocity_z': gps_measurements['AYTVEL02'],
})

#test
#print(df)


def ecef_to_eci(ecef_position, ecef_velocity, time):
    # add units
    pos = ecef_position * u.m
    vel = ecef_velocity * u.m / u.s

    now = Time(time)

    # convert coordinate to ECI
    itrs = coord.ITRS(x=pos[0], y=pos[1], z=pos[2],
                      v_x=vel[0], v_y=vel[1], v_z=vel[2],
                      representation_type='cartesian', differential_type='cartesian', obstime=now)
    gcrs = itrs.transform_to(coord.GCRS(obstime=now))

    pos_eci = gcrs.cartesian.xyz  # m
    pos_eci = pos_eci.value  # extract the values with units handled
    pos_eci = [float(p) for p in pos_eci]

    vel_eci = gcrs.velocity.d_xyz.to(u.m / u.s)  # ensure units are correct
    vel_eci = vel_eci.value  # extract the values with units handled
    vel_eci = [float(v) for v in vel_eci]

    return pos_eci, vel_eci

# ----------------------#
# convert gps positions #
# ----------------------#

positions_eci = []
velocities_eci = []

# Convert DataFrame columns to lists of tuples
gps_positions = df[['position_x', 'position_y', 'position_z']].values.tolist()
gps_velocities = df[['velocity_x', 'velocity_y', 'velocity_z']].values.tolist()
gps_times = df['time'].tolist()

for i in range(len(gps_positions)):
    pos_eci, vel_eci = ecef_to_eci(gps_positions[i], gps_velocities[i], gps_times[i])
    positions_eci.append(pos_eci)
    velocities_eci.append(vel_eci)

#print(gps_times)
#print(positions_eci)
#print(velocities_eci)

combined_data = []

for i in range(len(gps_times)):
   combined_data.append((gps_times[i], positions_eci[i], velocities_eci[i]))
#print(combined_data)


def flatten_data(data):
    return np.array([[*position, *velocity] for _, position, velocity in data])

# Flatten the data
flattened_data = flatten_data(combined_data)

print(len(flattened_data))
#print(combined_data)
#print(flattened_data)

scaler= StandardScaler().fit(flattened_data)
df_data_scaled= scaler.transform(flattened_data)

#training data
trainX=[]
#test data
trainY= []

#should predict one orbit
n_future= 90
n_past= 360

for i in range(n_past, len(df_data_scaled) - n_future+1):
    trainX.append(df_data_scaled[i-n_past:i, 0:flattened_data.shape[1]])
    trainY.append(df_data_scaled[i+n_future -1: i+n_future,0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}'.format(trainX.shape))
print('trainY shape == {}'.format(trainY.shape))

model = Sequential()
model.add(LSTM(128, activation="relu", input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(64, activation="relu", return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss= 'mse')
model.summary()

history = model.fit(trainX, trainY, epochs=50, batch_size=16, validation_split=0.2)

plt.plot(history.history['loss'], label= "Training loss")
plt.plot(history.history['val_loss'], label= "Validation loss")
plt.legend()
plt.show()

