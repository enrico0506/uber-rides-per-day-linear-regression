import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Read the CSV file
df = pd.read_csv('uber-raw-data-sep14.csv')

# Convert the 'Date/Time' column to datetime
df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

# Retain only the date part by extracting from the 'Date/Time' column
df['Formatted Date'] = df['Date/Time'].dt.date  

# GROUP BY DAY
rides_per_day = df.groupby('Formatted Date').size().reset_index(name='Number of Rides')

# Convert 'Date' to datetime
rides_per_day['Date'] = pd.to_datetime(rides_per_day['Formatted Date'])

# Sort the DataFrame by date
rides_per_day = rides_per_day.sort_values(by='Date')

# Create additional features for training
rides_per_day['DayOfWeek'] = rides_per_day['Date'].dt.dayofweek
rides_per_day['Month'] = rides_per_day['Date'].dt.month

# Define features and target variable
X_sept = rides_per_day[['DayOfWeek', 'Month']]
y_sept = rides_per_day[['Number of Rides']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sept, y_sept, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate future dates for October
future_dates_october = [pd.Timestamp('2014-10-01') + timedelta(days=x) for x in range(31)]

# Create a DataFrame for future dates
future_df = pd.DataFrame({'Date': future_dates_october})

# Create features for future dates
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['Month'] = future_df['Date'].dt.month

# Prepare X for predictions using the same features
future_X = future_df[['DayOfWeek', 'Month']]

# Make prediction for October
future_prediction = model.predict(future_X)

# Add predictions to the future DataFrame
future_df['Predicted Rides'] = future_prediction

# Visualize predictions and historical data
plt.figure(figsize=(14, 6))
plt.plot(rides_per_day['Date'], rides_per_day['Number of Rides'], label='Historical Rides', marker='o')
plt.plot(future_df['Date'], future_df['Predicted Rides'], label='Predicted Rides for October', linestyle='--', marker='x')
plt.title('Historical and Predicted Uber Rides')
plt.xlabel('Date')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()



'''
Plot and predict for each hour
'''

df_hours = df
df_hours['Date/Time'] = pd.to_datetime(df_hours['Date/Time'], errors='coerce')

rides_per_hour = df.groupby(df['Date/Time'].dt.floor('H')).size().reset_index(name='Number of Rides')

# Convert 'Date' to datetime
rides_per_hour['Date'] = pd.to_datetime(rides_per_hour['Date/Time'])

# Create additional features for training
rides_per_hour['DayOfWeek'] = rides_per_hour['Date'].dt.dayofweek
rides_per_hour['Hour'] = rides_per_hour['Date'].dt.hour
rides_per_hour['Month'] = rides_per_hour['Date'].dt.month

# Define features and target variable
X_sept = rides_per_hour[['DayOfWeek', 'Hour', 'Month']]
y_sept = rides_per_hour[['Number of Rides']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sept, y_sept, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate future timestamps for each hour of October
future_dates_october = [pd.Timestamp('2014-10-01') + timedelta(days=x) + timedelta(hours=h) 
                         for x in range(31) for h in range(24)]

# Create a DataFrame for future timestamps
future_df = pd.DataFrame({'Date': future_dates_october})

# Create features for future timestamps
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['Hour'] = future_df['Date'].dt.hour
future_df['Month'] = future_df['Date'].dt.month

# Prepare X for predictions
future_X = future_df[['DayOfWeek', 'Hour', 'Month']]

# Make prediction for October
future_prediction = model.predict(future_X)

# Add predictions to the future DataFrame
future_df['Predicted Rides'] = future_prediction

# Visualize predictions and historical data
plt.figure(figsize=(14, 6))
plt.plot(rides_per_hour['Date'], rides_per_hour['Number of Rides'], label='Historical Rides', marker='o', alpha=0.5)
plt.plot(future_df['Date'], future_df['Predicted Rides'], label='Predicted Rides for October', linestyle='--', marker='x')
plt.title('Historical and Predicted Uber Rides for Each Hour in October')
plt.xlabel('Date and Hour')
plt.ylabel('Number of Rides')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.show()