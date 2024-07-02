import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def convert_columns_to_float(df):
    for col in df.columns:
        print(f"Converting column {col} to float")
        df[col] = df[col].apply(
            lambda x: np.mean([float(i) for i in str(x).split(',') if is_float(i)]) if isinstance(x, str) else x)
        if df[col].isnull().any():
            print(f"Warning: NaN values detected in column {col} after conversion.")
    df = df.dropna()
    print(f"DataFrame after conversion to float:\n{df.head()}")
    return df


def clean_data(df):
    df = df.replace(['', 'NA', 'NaN'], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def load_and_transform_data():
    csv_files = {
        'car_telemetry': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/carTelemetry.csv',
        'car_damage': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/carDamage.csv',
        'car_setups': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/carSetups.csv',
        'car_status': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/carStatus.csv',
        'events': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/events.csv',
        'lap_data': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/lapData.csv',
        'motion': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/motion.csv',
        'tyre_sets': '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/tyreSets.csv'
    }

    car_telemetry = pd.read_csv(csv_files['car_telemetry'])
    car_damage = pd.read_csv(csv_files['car_damage'])
    car_setups = pd.read_csv(csv_files['car_setups'])
    car_status = pd.read_csv(csv_files['car_status'])
    events = pd.read_csv(csv_files['events'])
    lap_data = pd.read_csv(csv_files['lap_data'])
    motion = pd.read_csv(csv_files['motion'])
    tyre_sets = pd.read_csv(csv_files['tyre_sets'])

    print(f"Initial car_telemetry data:\n{car_telemetry.head()}")

    car_telemetry = clean_data(car_telemetry)
    car_damage = clean_data(car_damage)
    car_setups = clean_data(car_setups)
    car_status = clean_data(car_status)
    events = clean_data(events)
    lap_data = clean_data(lap_data)
    motion = clean_data(motion)
    tyre_sets = clean_data(tyre_sets)

    print(f"Cleaned car_telemetry data:\n{car_telemetry.head()}")

    car_telemetry = convert_columns_to_float(car_telemetry)
    car_damage = convert_columns_to_float(car_damage)
    car_setups = convert_columns_to_float(car_setups)
    car_status = convert_columns_to_float(car_status)
    lap_data = convert_columns_to_float(lap_data)
    motion = convert_columns_to_float(motion)
    tyre_sets = convert_columns_to_float(tyre_sets)

    if car_telemetry.empty:
        raise ValueError("Il dataframe car_telemetry è vuoto dopo la pulizia e la conversione.")
    if car_damage.empty:
        raise ValueError("Il dataframe car_damage è vuoto dopo la pulizia e la conversione.")
    if car_setups.empty:
        raise ValueError("Il dataframe car_setups è vuoto dopo la pulizia e la conversione.")
    if car_status.empty:
        raise ValueError("Il dataframe car_status è vuoto dopo la pulizia e la conversione.")
    if lap_data.empty:
        raise ValueError("Il dataframe lap_data è vuoto dopo la pulizia e la conversione.")
    if motion.empty:
        print("Warning: motion DataFrame is empty.")
    if tyre_sets.empty:
        print("Warning: tyre_sets DataFrame is empty.")

    scaler = StandardScaler()
    car_telemetry_transformed = csr_matrix(scaler.fit_transform(car_telemetry))
    car_damage_transformed = csr_matrix(scaler.fit_transform(car_damage))
    car_setups_transformed = csr_matrix(scaler.fit_transform(car_setups))
    car_status_transformed = csr_matrix(scaler.fit_transform(car_status))
    lap_data_transformed = csr_matrix(scaler.fit_transform(lap_data))
    motion_transformed = csr_matrix(scaler.fit_transform(motion)) if not motion.empty else csr_matrix(
        (0, len(motion.columns)))
    tyre_sets_transformed = csr_matrix(scaler.fit_transform(tyre_sets)) if not tyre_sets.empty else csr_matrix(
        (0, len(tyre_sets.columns)))

    car_telemetry_transformed_df = pd.DataFrame(car_telemetry_transformed.toarray(), columns=car_telemetry.columns)
    car_telemetry_transformed_df.to_csv(
        '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/transformed_carTelemetry.csv',
        index=False
    )

    return car_telemetry_transformed_df


def train_and_evaluate_model(car_telemetry_transformed_df):
    X = car_telemetry_transformed_df.drop('m_speed', axis=1)
    y = car_telemetry_transformed_df['m_speed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    joblib.dump(model,
                '/src/data/models/linear_regression_model.pkl')


def main():
    car_telemetry_transformed_df = load_and_transform_data()
    train_and_evaluate_model(car_telemetry_transformed_df)


if __name__ == "__main__":
    main()