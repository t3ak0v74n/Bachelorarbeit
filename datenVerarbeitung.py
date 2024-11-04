from tmApiClient import TmArchiveGet
import pandas as pd
import pickle
from datetime import datetime, timezone, timedelta
from astropy import units as u
from astropy import coordinates as coord
from astropy.time import Time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json



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

# Parameters
window_size = 90  # Number of time steps in the input
target_size = 10  # Number of time steps to predict
stride = 10       # Step size for overlapping sequences

# Create sequences and targets
def create_sequences(data, window_size, target_size, stride):
    sequences = []
    targets = []
    for start in range(0, len(data) - window_size - target_size + 1, stride):
        sequence = data[start:start + window_size]
        target = data[start + window_size:start + window_size + target_size]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Generate sequences and targets
X, y = create_sequences(flattened_data, window_size, target_size, stride)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X for the LSTM: (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 6))  # 6 features: x, y, z, vx, vy, vz
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 6))

# Build the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=(window_size, 6)),
    keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Dense(target_size * 6),  # Output layer to predict next 90 steps with 6 features each
    keras.layers.Reshape((target_size, 6))  # Reshape output to (90, 6)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model without EarlyStopping
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test)

# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()