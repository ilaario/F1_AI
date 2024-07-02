import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Define drivers and tracks
drivers = ['VER', 'PER', 'HAM', 'RUS', 'NOR', 'PIA', 'SAI', 'LEC', 'ALO', 'STR', 'GAS', 'OCO', 'TSU', 'BOT', 'ZHO', 'MAG', 'HUL', 'ALB', 'SAR']
tracks = ['Imola', 'Monza', 'Sochi', 'Austin', 'Mexico', 'Interlagos', 'Melbourne', 'Monte Carlo', 'Baku', 'Montreal', 'Silverstone', 'Spielberg', 'Budapest', 'Spa', 'Zandvoort', 'Monaco', 'Jeddah', 'Suzuka', 'Abu Dhabi', 'Sakhir']

base_path = '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/real_datas/2023'
save_path = '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/clustered_datas'

# Initialize an empty DataFrame to store all combined data
all_data = pd.DataFrame()

for track in tracks:
    for driver in drivers:
        if os.path.exists(os.path.join(save_path, f"{driver}_{track}_clustered.csv")):
            print(f"Skipping {driver} at {track}, data already clustered.")
            continue
        try:
            # Load telemetry and position data
            telemetry_file = os.path.join(base_path, track, f"{driver}_telemetry.csv")
            position_file = os.path.join(base_path, track, f"{driver}_position.csv")

            telemetry_data = pd.read_csv(telemetry_file)
            print(f'{driver}_telemetry.csv from {track} loaded successfully')
            position_data = pd.read_csv(position_file)
            print(f'{driver}_position.csv from {track} loaded successfully')

            # Ensure consistent datetime format
            telemetry_data['Date'] = pd.to_datetime(telemetry_data['Date'])
            position_data['Date'] = pd.to_datetime(position_data['Date'])

            # Merge telemetry and position data
            merged_data = pd.merge_asof(telemetry_data.sort_values('Date'),
                                        position_data.sort_values('Date'),
                                        on='Date',
                                        direction='nearest')

            # Append to the all_data DataFrame
            # Perform clustering on the combined data
            features = ['RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'X', 'Y', 'Z']
            merged_data = merged_data.dropna(subset=features)
            X = merged_data[features]

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            merged_data['Cluster'] = kmeans.fit_predict(X)

            # Save the clustered data
            path = os.path.join(save_path, f'{driver}_{track}_clustered.csv')
            merged_data.to_csv(path, index=False)
            print(f'Clustering of Telemetry and Position Data for {driver} in {track} complete.')
        except Exception as e:
            print(f"Error processing {driver} at {track}: {e}")


print("----------------------------------------------------------")
cluster_path = '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/clustered_datas'

# Initialize an empty DataFrame to store all combined data
all_data = pd.DataFrame()

for track in tracks:
    for driver in drivers:
        try:
            # Load clustered data
            clustered_file = os.path.join(cluster_path, f"{driver}_{track}_clustered.csv")
            clustered_data = pd.read_csv(clustered_file)

            # Append to the all_data DataFrame
            all_data = pd.concat([all_data, clustered_data], ignore_index=True)
            print(f'{driver}_{track}_clustered.csv loaded successfully')
        except Exception as e:
            print(f"Error processing {driver} at {track}: {e}")

# Define features and target variable
features = ['RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'X', 'Y', 'Z']
target = 'Cluster'

# Drop any rows with missing values
all_data = all_data.dropna(subset=features + [target])

# Split the data into training and testing sets
X = all_data[features]
y = all_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/models/action_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

