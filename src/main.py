import math
import os
import json
import sys

import pandas as pd
import joblib
import numpy as np
import socket
import matplotlib.pyplot as plt
import time
import subprocess
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import StandardScaler


from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ProgbarLogger, Callback
from sklearn.metrics import mean_squared_error
from io import StringIO
from threading import Thread, Event

from torch.utils.data import TensorDataset, DataLoader


def print_progress_bar(iteration, total, length=50):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent}% Complete')
    sys.stdout.flush()


# Configurazione del socket TCP per ricevere i dati
TCP_IP = "127.0.0.1"
TCP_PORT = 41234
conn = None
records = []
predictions = []
server_ready = Event()
scaler_file_path = os.path.join(os.getcwd(), "scaler.pkl")


# Funzione per avviare il server TCP
def start_tcp_server():
    global conn
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(1)
    print("Waiting for connection...")
    server_ready.set()  # Indica che il server è pronto
    conn, addr = sock.accept()
    print(f"Connected by {addr}")

def convert_bigint(data):
    if isinstance(data, dict):
        return {key: convert_bigint(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_bigint(element) for element in data]
    elif isinstance(data, str) and data.endswith("n"):
        return int(data[:-1])
    else:
        return data

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} ended. Logs: {logs}")

def load_scaler(file_path):
    try:
        scaler = joblib.load(file_path)
        print("Scaler caricato correttamente.")
        return scaler
    except PermissionError as e:
        print(f"Errore di permessi: {e}")
    except FileNotFoundError as e:
        print(f"File non trovato: {e}")
    except Exception as e:
        print(f"Errore durante il caricamento dello scaler: {e}")
        return None

def custom_loss(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)

    # Penalità per errori nelle sezioni critiche (es. velocità troppo alta)
    critical_threshold = 150.0  # Esempio di soglia critica per la velocità
    speed_penalty = torch.mean(
        torch.where(y_true[:, 0] > critical_threshold, (y_true[:, 0] - y_pred[:, 0]) ** 2, torch.tensor(0.0).to(y_true.device))
    )

    # Penalità per surriscaldamento gomme
    tyre_temp_threshold = 100.0  # Soglia di temperatura delle gomme (esempio)
    tyre_temp_penalty = torch.mean(
        torch.where(y_true[:, 20:24] > tyre_temp_threshold, (y_true[:, 20:24] - y_pred[:, 20:24]) ** 2, torch.tensor(0.0).to(y_true.device))
    )

    # Penalità per sterzata eccessiva
    steer_penalty = torch.mean((y_true[:, 2] - y_pred[:, 2]) ** 2)

    # Penalità per valori negativi nei risultati
    negative_penalty = torch.mean(torch.where(y_pred < 0, y_pred ** 2, torch.tensor(0.0).to(y_pred.device)))

    return mse + speed_penalty + tyre_temp_penalty + steer_penalty + negative_penalty

def load_and_normalize_json():
    # Carica il file JSON riga per riga
    event_data = []
    car_telemetry_data = []
    car_setups_data = []
    car_damage_data = []
    car_status_data = []
    tyre_sets_data = []
    lap_data = []
    motion_data = []

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/events.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                event_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di events.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/tyreSets.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                tyre_sets_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di tyre_sets.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI '
              'Project/src/data/carTelemetry.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                car_telemetry_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di telemetry_data.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/carSetups.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                car_setups_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di car_setup.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/carDamage.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                car_damage_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di car_damage.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/carStatus.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                car_status_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di car_status.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/motion.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                motion_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di motion_data.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/lapData.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                lap_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di lap_data.json: {e}")
                continue

    # Normalizza ogni lista di record JSON in DataFrame separati
    event_df = pd.json_normalize(event_data)
    car_telemetry_df = pd.json_normalize(car_telemetry_data, 'm_carTelemetryData', ['m_header'])
    car_setups_df = pd.json_normalize(car_setups_data, 'm_carSetups', ['m_header'])
    car_damage_df = pd.json_normalize(car_damage_data, 'm_carDamageData', ['m_header'])
    car_status_df = pd.json_normalize(car_status_data, 'm_carStatusData', ['m_header'])
    tyre_sets_df = pd.json_normalize(tyre_sets_data, 'm_tyreSetData', ['m_header'])
    motion_df = pd.json_normalize(motion_data, 'm_carMotionData', ['m_header'])
    lap_df = pd.json_normalize(lap_data, 'm_lapData', ['m_header'])

    # Gestione delle colonne con array (esempio per car_telemetry_df)
    car_telemetry_df = pd.concat([
        car_telemetry_df.drop(
            ['m_brakesTemperature', 'm_tyresSurfaceTemperature', 'm_tyresInnerTemperature', 'm_tyresPressure',
             'm_surfaceType'], axis=1),
        car_telemetry_df['m_brakesTemperature'].apply(pd.Series).rename(lambda x: f'm_brakesTemperature_{x}',
                                                                        axis=1),
        car_telemetry_df['m_tyresSurfaceTemperature'].apply(pd.Series).rename(
            lambda x: f'm_tyresSurfaceTemperature_{x}', axis=1),
        car_telemetry_df['m_tyresInnerTemperature'].apply(pd.Series).rename(
            lambda x: f'm_tyresInnerTemperature_{x}', axis=1),
        car_telemetry_df['m_tyresPressure'].apply(pd.Series).rename(lambda x: f'm_tyresPressure_{x}', axis=1),
        car_telemetry_df['m_surfaceType'].apply(pd.Series).rename(lambda x: f'm_surfaceType_{x}', axis=1)
    ], axis=1)

    # Unisci tutti i DataFrame in uno unico
    df = pd.concat([event_df, car_telemetry_df, car_setups_df, car_damage_df, car_status_df, tyre_sets_df, motion_df, lap_df],
                   ignore_index=True, sort=False)

    # Visualizza i nomi delle colonne per verifica
    print(df.columns)

    return df


def normalize_records(records):
    if not records:
        return records

    # Trova tutte le chiavi presenti in tutti i record
    all_keys = set()
    for record in records:
        all_keys.update(record.keys())

    # Assicura che tutti i record abbiano le stesse chiavi
    for record in records:
        for key in all_keys:
            if key not in record:
                record[key] = None

    data = {key: [] for key in all_keys}

    for record in records:
        for key in record:
            data[key].append(record[key])

    for key in data:
        if key != "sessionTime" and data[key][0] is not None:  # Evita di normalizzare il tempo della sessione
            min_val = min(x for x in data[key] if x is not None)
            max_val = max(x for x in data[key] if x is not None)
            if min_val == max_val:  # Evita divisione per zero
                data[key] = [0.5 if x is not None else None for x in data[key]]
            else:
                data[key] = [(x - min_val) / (max_val - min_val) if x is not None else None for x in data[key]]

    for i, record in enumerate(records):
        for key in record:
            if key != "sessionTime" and data[key][i] is not None:
                record[key] = data[key][i]

    return records

class CarSimulator:
    def __init__(self):
        self.speed = 0.0
        self.rpm = 4500
        self.gear = 1
        self.position = [0.0, 0.0]  # posizione X, Y
        self.direction = 0.0  # angolo di direzione
        self.max_rpm = 14000
        self.max_speed = 350
        self.gear_ratios = [0, 88.3, 72.3, 60.1, 51.2, 44.9, 39.7, 35.7]  # Esempio di rapporti del cambio

    def update(self, throttle, brake, clutch, gear, steer):
        # Aggiorna la marcia
        if 0 < gear < len(self.gear_ratios):
            self.gear = gear

        # Calcola la nuova velocità e RPM in base all'accelerazione e alla marcia
        if throttle > 0:
            acceleration = throttle * 10.68  # Acceleration rate
            self.speed += acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed

        if brake > 0:
            deceleration = brake * 39  # Deceleration rate
            self.speed -= deceleration
            if self.speed < 0:
                self.speed = 0

        # Calcola gli RPM in base alla velocità e alla marcia
        if self.gear > 0:
            self.rpm = self.speed * self.gear_ratios[self.gear]
            if self.rpm > self.max_rpm:
                self.rpm = self.max_rpm

        # Aggiorna la direzione della macchina basata sull'angolo di sterzata
        steer_radians = np.deg2rad(steer)
        self.direction += steer_radians * 0.01  # Adjust the steering sensitivity

        # Aggiorna la posizione della macchina
        self.position[0] += self.speed * np.cos(self.direction)
        self.position[1] += self.speed * np.sin(self.direction)

        return self.get_state()

    def get_state(self):
        return {
            "speed": self.speed,
            "rpm": self.rpm,
            "gear": self.gear,
            "direction": self.direction,
            "position_x": self.position[0],
            "position_y": self.position[1]
        }


def train_model():

    # start_json_node_client()

    # Carica e normalizza i dati JSON
    df = load_and_normalize_json()

    simulator = CarSimulator()
    simulation_data = []

    # Simulazione del giro di pista
    for _, record in df.iterrows():
        throttle = record['m_throttle']
        brake = record['m_brake']
        clutch = record['m_clutch']
        gear = record['m_gear']
        steer_angle = record['m_steer']  # Assumi che l'angolo di sterzata sia in gradi
        if not math.isnan(steer_angle) or not math.isnan(throttle) or not math.isnan(brake) or not math.isnan(clutch) or not math.isnan(gear):
            print("Throttle: ", throttle)
            print("Brake: ", brake)
            print("Clutch: ", clutch)
            gear = int(gear)
            print("Gear: ", gear)
            print("Steer angle: ", steer_angle)
            state = simulator.update(throttle, brake, clutch, gear, steer_angle)
            simulation_data.append(state)
            print("State: ", state)
            print("------------------------------------------------------------")

    # Salvataggio dei dati simulati in un file CSV
    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.to_csv('simulation_data.csv', index=False)

    print("DataFrame originale:\n", df.head())

    # Controllo dei valori NaN nel DataFrame
    print("\nNumero di valori NaN per colonna:\n", df.isna().sum())

    # Sostituzione dei NaN con 0
    df.fillna(0, inplace=True)

    # Controllo del DataFrame dopo la sostituzione dei NaN
    print("\nDataFrame dopo la sostituzione dei NaN:\n", df.head())

    # Selezione delle colonne di input e output
    input_columns = ['m_worldPositionX', 'm_worldPositionY', 'm_worldPositionZ',
                     'm_worldVelocityX', 'm_worldVelocityY', 'm_worldVelocityZ',
                     'm_worldForwardDirX', 'm_worldForwardDirY', 'm_worldForwardDirZ',
                     'm_worldRightDirX', 'm_worldRightDirY', 'm_worldRightDirZ',
                     'm_gForceLateral', 'm_gForceLongitudinal', 'm_gForceVertical',
                     'm_yaw', 'm_pitch', 'm_roll', 'm_engineRPM']

    output_columns = ['m_throttle', 'm_brake', 'm_clutch', 'm_gear']

    # Normalizzazione dei dati
    scaler_X = StandardScaler()
    df[input_columns] = scaler_X.fit_transform(df[input_columns])

    # Definizione dell'ambiente Gym
    class RacingEnv(gym.Env):
        def __init__(self, df):
            super(RacingEnv, self).__init__()
            self.df = df
            self.current_step = 0
            self.action_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]),
                                               dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,),
                                                    dtype=np.float32)  # Speed, RPM, Gear, Position

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            return self.df.iloc[self.current_step].values, {}

        def step(self, action):
            throttle, brake, clutch, gear = action
            self.current_step += 1
            if self.current_step >= len(self.df):
                self.current_step = 0  # Ricomincia dall'inizio per continuare l'addestramento

            state = self.df.iloc[self.current_step].values

            reward = self.calculate_reward(state, action)

            done = self.current_step == len(self.df) - 1

            return state, reward, done, False, {}

        def calculate_reward(self, state, action):
            speed, rpm, gear, position_x, position_y = state

            reward = 0
            optimal_rpm_low = 10000.0
            optimal_rpm_high = 13500.0

            # Penalità per RPM non ottimali
            if optimal_rpm_low <= rpm <= optimal_rpm_high:
                reward += 1
            else:
                reward -= 0.1

            # Penalità per cambi di marcia non ottimali
            if gear > 0 and (rpm < optimal_rpm_low or rpm > optimal_rpm_high):
                reward -= 0.5

            # Penalità per velocità troppo alta
            if speed > 350:
                reward -= 1

            return reward

    # Creazione dell'ambiente con il wrapper VecMonitor
    env = DummyVecEnv([lambda: RacingEnv(df)])
    env = VecMonitor(env)

    # Addestramento del modello PPO
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    # Valutazione del modello
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Salvataggio del modello
    model.save("ppo_racing")

    # Caricamento del modello
    model = PPO.load("ppo_racing")

    # Predizione delle azioni
    obs = env.reset()
    for i in range(len(df)):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f"Step {i}, Action: {action}, Reward: {rewards}")

    print("Fine del programma.")
    exit(0)

def preprocess_and_predict(data, model, scaler):
    features = pd.DataFrame(data)
    features_scaled = scaler.transform(features)
    features_scaled = features_scaled.reshape(
        (features_scaled.shape[0], 1, features_scaled.shape[1])
    )  # Reshape per LSTM
    predictions = model.predict(features_scaled)
    return predictions

def read_process_output(proc):
    while True:
        output = proc.stdout.readline()
        if proc.poll() is not None and output == "":
            break
        if output:
            print(output.strip())

def collect_data():
    global conn, records
    buffer = ""
    start_time = time.time()
    while conn is None:
        time.sleep(0.1)  # Attendi che la connessione sia stabilita
    try:
        while time.time() - start_time < 10:  # 30 minuti
            data = conn.recv(4096).decode("utf-8")
            if not data:
                continue
            buffer += data
            try:
                packets = buffer.split("\n")
                buffer = packets.pop()
                for packet in packets:
                    if not packet:
                        continue
                    json_packet = json.loads(packet.strip())
                    #json_packet = convert_bigint(json_packet)
                    if json_packet["m_header"]["m_packetId"] == 0: # motion packet ricevuto
                        print("Motion packet ricevuto.")
                        session_time = json_packet["m_header"]["m_sessionTime"]
                        for car_motion in json_packet["m_carMotionData"]:
                            record = {
                                "sessionTime": session_time,
                                "worldPositionX": car_motion["m_worldPositionX"],
                                "worldPositionY": car_motion["m_worldPositionY"],
                                "worldPositionZ": car_motion["m_worldPositionZ"],
                                "worldVelocityX": car_motion["m_worldVelocityX"],
                                "worldVelocityY": car_motion["m_worldVelocityY"],
                                "worldVelocityZ": car_motion["m_worldVelocityZ"],
                                "worldForwardDirX": car_motion["m_worldForwardDirX"],
                                "worldForwardDirY": car_motion["m_worldForwardDirY"],
                                "worldForwardDirZ": car_motion["m_worldForwardDirZ"],
                                "worldRightDirX": car_motion["m_worldRightDirX"],
                                "worldRightDirY": car_motion["m_worldRightDirY"],
                                "worldRightDirZ": car_motion["m_worldRightDirZ"],
                                "gForceLateral": car_motion["m_gForceLateral"],
                                "gForceLongitudinal": car_motion["m_gForceLongitudinal"],
                                "gForceVertical": car_motion["m_gForceVertical"],
                                "yaw": car_motion["m_yaw"],
                                "pitch": car_motion["m_pitch"],
                                "roll": car_motion["m_roll"],
                            }
                            records.append(record)
                    elif json_packet["m_header"]["m_packetId"] == 2: # lap data package
                        print("Lap data package ricevuto.")
                        session_time = json_packet["m_header"]["m_sessionTime"]
                        for lap_data in json_packet["m_lapData"]:
                            record = {
                                "sessionTime": session_time,
                                "lastLapTime": lap_data["m_lastLapTimeInMS"],
                                "currentLapTime": lap_data["m_currentLapTimeInMS"],
                                "sector1Time": lap_data["m_sector1TimeInMS"],
                                "sector2Time": lap_data["m_sector2TimeInMS"],
                                "lapDistance": lap_data["m_lapDistance"],
                                "totalDistance": lap_data["m_totalDistance"],
                                "safetyCarDelta": lap_data["m_safetyCarDelta"],
                                "carPosition": lap_data["m_carPosition"],
                                "currentLapNum": lap_data["m_currentLapNum"],
                                "pitStatus": lap_data["m_pitStatus"],
                                "sector": lap_data["m_sector"],
                                "currentLapInvalid": lap_data["m_currentLapInvalid"],
                                "penalties": lap_data["m_penalties"],
                                "gridPosition": lap_data["m_gridPosition"],
                                "driverStatus": lap_data["m_driverStatus"],
                                "resultStatus": lap_data["m_resultStatus"]
                            }
                            records.append(record)
                    elif json_packet["m_header"]["m_packetId"] == 3: # event package
                        print("Event package ricevuto.")
                        # check only DRSE and DRSD events
                        if json_packet["m_eventStringCode"] == "DRSE" or json_packet["m_eventStringCode"] == "DRSD":
                            session_time = json_packet["m_header"]["m_sessionTime"]
                            record = {
                                "sessionTime": session_time,
                                "eventStringCode": json_packet["m_eventStringCode"]
                            }
                            records.append(record)
                    elif json_packet["m_header"]["m_packetId"] == 5: # car setups package
                        print("Car setups package ricevuto.")
                        session_time = json_packet["m_header"]["m_sessionTime"]
                        for car_setup in json_packet["m_carSetups"]:
                            record = {
                                "sessionTime": session_time,
                                "frontWing": car_setup["m_frontWing"],
                                "rearWing": car_setup["m_rearWing"],
                                "onThrottle": car_setup["m_onThrottle"],
                                "offThrottle": car_setup["m_offThrottle"],
                                "frontCamber": car_setup["m_frontCamber"],
                                "rearCamber": car_setup["m_rearCamber"],
                                "frontToe": car_setup["m_frontToe"],
                                "rearToe": car_setup["m_rearToe"],
                                "frontSuspension": car_setup["m_frontSuspension"],
                                "rearSuspension": car_setup["m_rearSuspension"],
                                "frontAntiRollBar": car_setup["m_frontAntiRollBar"],
                                "rearAntiRollBar": car_setup["m_rearAntiRollBar"],
                                "frontSuspensionHeight": car_setup["m_frontSuspensionHeight"],
                                "rearSuspensionHeight": car_setup["m_rearSuspensionHeight"],
                                "brakePressure": car_setup["m_brakePressure"],
                                "brakeBias": car_setup["m_brakeBias"],
                                "rearLeftTyrePressure": car_setup["m_rearLeftTyrePressure"],
                                "rearRightTyrePressure": car_setup["m_rearRightTyrePressure"],
                                "frontLeftTyrePressure": car_setup["m_frontLeftTyrePressure"],
                                "frontRightTyrePressure": car_setup["m_frontRightTyrePressure"],
                                "ballast": car_setup["m_ballast"],
                                "fuelLoad": car_setup["m_fuelLoad"],
                            }
                            records.append(record)
                    elif json_packet["m_header"]["m_packetId"] == 6:
                        print("Car telemetry package ricevuto.")
                        session_time = json_packet["m_header"]["m_sessionTime"]
                        for car_telemetry in json_packet["m_carTelemetryData"]:
                            record = {
                                "sessionTime": session_time,
                                "speed": car_telemetry["m_speed"],
                                "throttle": car_telemetry["m_throttle"],
                                "steer": car_telemetry["m_steer"],
                                "brake": car_telemetry["m_brake"],
                                "clutch": car_telemetry["m_clutch"],
                                "gear": car_telemetry["m_gear"],
                                "engineRPM": car_telemetry["m_engineRPM"],
                                "drs": car_telemetry["m_drs"],
                                "revLightsPercent": car_telemetry["m_revLightsPercent"],
                                "revLightsBitValue": car_telemetry[
                                    "m_revLightsBitValue"
                                ],
                                "brakesTemperature_FL": car_telemetry[
                                    "m_brakesTemperature"
                                ][0],
                                "brakesTemperature_FR": car_telemetry[
                                    "m_brakesTemperature"
                                ][1],
                                "brakesTemperature_RL": car_telemetry[
                                    "m_brakesTemperature"
                                ][2],
                                "brakesTemperature_RR": car_telemetry[
                                    "m_brakesTemperature"
                                ][3],
                                "tyresSurfaceTemperature_FL": car_telemetry[
                                    "m_tyresSurfaceTemperature"
                                ][0],
                                "tyresSurfaceTemperature_FR": car_telemetry[
                                    "m_tyresSurfaceTemperature"
                                ][1],
                                "tyresSurfaceTemperature_RL": car_telemetry[
                                    "m_tyresSurfaceTemperature"
                                ][2],
                                "tyresSurfaceTemperature_RR": car_telemetry[
                                    "m_tyresSurfaceTemperature"
                                ][3],
                                "tyresInnerTemperature_FL": car_telemetry[
                                    "m_tyresInnerTemperature"
                                ][0],
                                "tyresInnerTemperature_FR": car_telemetry[
                                    "m_tyresInnerTemperature"
                                ][1],
                                "tyresInnerTemperature_RL": car_telemetry[
                                    "m_tyresInnerTemperature"
                                ][2],
                                "tyresInnerTemperature_RR": car_telemetry[
                                    "m_tyresInnerTemperature"
                                ][3],
                                "engineTemperature": car_telemetry[
                                    "m_engineTemperature"
                                ],
                                "tyresPressure_FL": car_telemetry["m_tyresPressure"][0],
                                "tyresPressure_FR": car_telemetry["m_tyresPressure"][1],
                                "tyresPressure_RL": car_telemetry["m_tyresPressure"][2],
                                "tyresPressure_RR": car_telemetry["m_tyresPressure"][3],
                                "surfaceType_FL": car_telemetry["m_surfaceType"][0],
                                "surfaceType_FR": car_telemetry["m_surfaceType"][1],
                                "surfaceType_RL": car_telemetry["m_surfaceType"][2],
                                "surfaceType_RR": car_telemetry["m_surfaceType"][3],
                            }
                            records.append(record)
                    elif json_packet["m_header"]["m_packetId"] == 7:
                        print("Car status package ricevuto.")
                        session_time = json_packet["m_header"]["m_sessionTime"]
                        for car_status in json_packet["m_carStatusData"]:
                            record = {
                                "sessionTime": session_time,
                                "tractionControl": car_status["m_tractionControl"],
                                "antiLockBrakes": car_status["m_antiLockBrakes"],
                                "fuelMix": car_status["m_fuelMix"],
                                "frontBrakeBias": car_status["m_frontBrakeBias"],
                                "pitLimiterStatus": car_status["m_pitLimiterStatus"],
                                "fuelInTank": car_status["m_fuelInTank"],
                                "fuelCapacity": car_status["m_fuelCapacity"],
                                "fuelRemainingLaps": car_status["m_fuelRemainingLaps"],
                                "maxRPM": car_status["m_maxRPM"],
                                "idleRPM": car_status["m_idleRPM"],
                                "maxGears": car_status["m_maxGears"],
                                "drsAllowed": car_status["m_drsAllowed"],
                                "drsActivationDistance": car_status["m_drsActivationDistance"],
                                "actualTyreCompound": car_status["m_actualTyreCompound"],
                                "visualTyreCompound": car_status["m_visualTyreCompound"],
                                "tyresAgeLaps": car_status["m_tyresAgeLaps"],
                                "vehicleFiaFlags": car_status["m_vehicleFiaFlags"],
                                "ersStoreEnergy": car_status["m_ersStoreEnergy"],
                                "ersDeployMode": car_status["m_ersDeployMode"],
                                "ersHarvestedThisLapMGUK": car_status["m_ersHarvestedThisLapMGUK"],
                                "ersHarvestedThisLapMGUH": car_status["m_ersHarvestedThisLapMGUH"],
                                "ersDeployedThisLap": car_status["m_ersDeployedThisLap"],
                            }
                            records.append(record)
                    elif json_packet["m_header"]["m_packetId"] == 10: # car damage package
                        print("Car damage package ricevuto.")
                        session_time = json_packet["m_header"]["m_sessionTime"]
                        for car_damage in json_packet["m_carDamageData"]:
                            record = {
                                "sessionTime": session_time,
                                "tyresWear_FL": car_damage["m_tyresWear"][0],
                                "tyresWear_FR": car_damage["m_tyresWear"][1],
                                "tyresWear_RL": car_damage["m_tyresWear"][2],
                                "tyresWear_RR": car_damage["m_tyresWear"][3],
                                "tyresDamage_FL": car_damage["m_tyresDamage"][0],
                                "tyresDamage_FR": car_damage["m_tyresDamage"][1],
                                "tyresDamage_RL": car_damage["m_tyresDamage"][2],
                                "tyresDamage_RR": car_damage["m_tyresDamage"][3],
                                "brakesDamage_FL": car_damage["m_brakesDamage"][0],
                                "brakesDamage_FR": car_damage["m_brakesDamage"][1],
                                "brakesDamage_RL": car_damage["m_brakesDamage"][2],
                                "brakesDamage_RR": car_damage["m_brakesDamage"][3],
                                "frontLeftWingDamage": car_damage["m_frontLeftWingDamage"],
                                "frontRightWingDamage": car_damage["m_frontRightWingDamage"],
                                "rearWingDamage": car_damage["m_rearWingDamage"],
                                "floorDamage": car_damage["m_floorDamage"],
                                "diffuserDamage": car_damage["m_diffuserDamage"],
                                "sidepodDamage": car_damage["m_sidepodDamage"],
                                "drsFault": car_damage["m_drsFault"],
                                "ersFault": car_damage["m_ersFault"],
                                "gearBoxDamage": car_damage["m_gearBoxDamage"],
                                "engineDamage": car_damage["m_engineDamage"],
                                "engineMGUHWear": car_damage["m_engineMGUHWear"],
                                "engineESWear": car_damage["m_engineESWear"],
                                "engineCEWear": car_damage["m_engineCEWear"],
                                "engineICEWear": car_damage["m_engineICEWear"],
                                "engineMGUKWear": car_damage["m_engineMGUKWear"],
                                "engineTCWear": car_damage["m_engineTCWear"],
                                "engineBlown": car_damage["m_engineBlown"],
                                "engineSeized": car_damage["m_engineSeized"]
                            }
                            records.append(record)
                    elif json_packet["m_header"]["m_packetId"] == 12: # tyre sets package
                        print("Tyre sets package ricevuto.")
                        session_time = json_packet["m_header"]["m_sessionTime"]
                        for tyre_set in json_packet["m_tyreSetData"]:
                            record = {
                                "sessionTime": session_time,
                                "tyreCompound": tyre_set["m_actualTyreCompound"],
                                "visualTyreCompound": tyre_set["m_visualTyreCompound"],
                                "wear": tyre_set["m_wear"],
                                "available": tyre_set["m_available"],
                                "recommendedSession": tyre_set["m_recommendedSession"],
                                "lifeSpan": tyre_set["m_lifeSpan"],
                                "usableLife": tyre_set["m_usableLife"],
                                "lapDeltaTime": tyre_set["m_lapDeltaTime"],
                                "fitted": tyre_set["m_fitted"]
                            }
                            records.append(record)

                    else:
                        print(f"Packet ID non gestito: {json_packet['m_header']['m_packetId']}")
                        continue

            except json.JSONDecodeError as e:
                print(f"Errore nella decodifica del JSON: {e}")
                continue

        stdout, stderr = proc.communicate()
        if stdout:
            print("STDOUT:", stdout)
        if stderr:
            print("STDERR:", stderr)

    except KeyboardInterrupt:
        print("Interrotto dall'utente durante la raccolta dati.")
    finally:
        proc.kill()
        print("Chiusura del programma di raccolta dati.")

    train_model(records)

def start_json_node_client():
    telemetry_script = "node src/telemetry_to_json.mjs"

    try:
        proc = subprocess.Popen(
            telemetry_script,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"Avviato script: {telemetry_script}")

        thread = Thread(target=read_process_output, args=(proc,))
        thread.start()

        time.sleep(5)

        # wait for 30 minutes and then kill the process
        total_seconds = 1 * 60

        for elapsed_seconds in range(total_seconds + 1):
            print_progress_bar(elapsed_seconds, total_seconds)
            time.sleep(1)  # Attende un secondo tra ogni iterazione
        print()  # Per andare a capo alla fine della barra di avanzamento

        proc.kill()
    except Exception as e:
        print(f"Errore nell'avvio dello script: {e}")
        exit(1)

def start_node_client():
    telemetry_script = "node src/telemetry_to_json.mjs"

    try:
        proc = subprocess.Popen(
            telemetry_script,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"Avviato script: {telemetry_script}")

        thread = Thread(target=read_process_output, args=(proc,))
        thread.start()
    except Exception as e:
        print(f"Errore nell'avvio dello script: {e}")
        exit(1)


model = None
scaler = None
if not os.path.exists("f1_deep_learning_model.h5"):
    print(
        "Modello non trovato. Raccolta dati per 30 minuti e addestramento del modello."
    )
    train_model()
else:
    model = tf.keras.models.load_model("f1_deep_learning_model.h5")
    scaler_file_path = "scaler.pkl"
    scaler = load_scaler(scaler_file_path)

    if scaler is None:
        # Se lo scaler non esiste, creane uno nuovo e salvalo
        scaler = StandardScaler()
        joblib.dump(scaler, scaler_file_path)
    scaler = joblib.load("scaler.pkl")
    print("Modello trovato. Inizio la raccolta dei dati per le predizioni.")

print("Inizio la raccolta dei dati per le predizioni.")

plt.ion()
fig, ax = plt.subplots()
(line1,) = ax.plot([], [], label="Valori Reali")
(line2,) = ax.plot([], [], label="Predizioni", linestyle="--")
ax.set_xlabel("Tempo")
ax.set_ylabel("Session Time")
ax.legend()

try:
    while True:
        start_time = time.time()
        while conn is None:
            time.sleep(0.1)  # Attendi che la connessione sia stabilita
        data = conn.recv(4096).decode("utf-8")
        if not data:
            print("Nessun dato ricevuto.")
            continue

        packets = data.split("\n")
        for packet in packets:
            if not packet:
                continue
            json_packet = json.loads(packet.strip())
            json_packet = convert_bigint(json_packet)
            if json_packet["m_header"]["m_packetId"] == 6:
                session_time = json_packet["m_header"]["m_sessionTime"]
                for car_telemetry in json_packet["m_carTelemetryData"]:
                    record = {
                        "sessionTime": session_time,
                        "speed": car_telemetry["m_speed"],
                        "throttle": car_telemetry["m_throttle"],
                        "steer": car_telemetry["m_steer"],
                        "brake": car_telemetry["m_brake"],
                        "clutch": car_telemetry["m_clutch"],
                        "gear": car_telemetry["m_gear"],
                        "engineRPM": car_telemetry["m_engineRPM"],
                        "drs": car_telemetry["m_drs"],
                        "revLightsPercent": car_telemetry["m_revLightsPercent"],
                        "revLightsBitValue": car_telemetry["m_revLightsBitValue"],
                        "brakesTemperature_FL": car_telemetry["m_brakesTemperature"][0],
                        "brakesTemperature_FR": car_telemetry["m_brakesTemperature"][1],
                        "brakesTemperature_RL": car_telemetry["m_brakesTemperature"][2],
                        "brakesTemperature_RR": car_telemetry["m_brakesTemperature"][3],
                        "tyresSurfaceTemperature_FL": car_telemetry[
                            "m_tyresSurfaceTemperature"
                        ][0],
                        "tyresSurfaceTemperature_FR": car_telemetry[
                            "m_tyresSurfaceTemperature"
                        ][1],
                        "tyresSurfaceTemperature_RL": car_telemetry[
                            "m_tyresSurfaceTemperature"
                        ][2],
                        "tyresSurfaceTemperature_RR": car_telemetry[
                            "m_tyresSurfaceTemperature"
                        ][3],
                        "tyresInnerTemperature_FL": car_telemetry[
                            "m_tyresInnerTemperature"
                        ][0],
                        "tyresInnerTemperature_FR": car_telemetry[
                            "m_tyresInnerTemperature"
                        ][1],
                        "tyresInnerTemperature_RL": car_telemetry[
                            "m_tyresInnerTemperature"
                        ][2],
                        "tyresInnerTemperature_RR": car_telemetry[
                            "m_tyresInnerTemperature"
                        ][3],
                        "engineTemperature": car_telemetry["m_engineTemperature"],
                        "tyresPressure_FL": car_telemetry["m_tyresPressure"][0],
                        "tyresPressure_FR": car_telemetry["m_tyresPressure"][1],
                        "tyresPressure_RL": car_telemetry["m_tyresPressure"][2],
                        "tyresPressure_RR": car_telemetry["m_tyresPressure"][3],
                        "surfaceType_FL": car_telemetry["m_surfaceType"][0],
                        "surfaceType_FR": car_telemetry["m_surfaceType"][1],
                        "surfaceType_RL": car_telemetry["m_surfaceType"][2],
                        "surfaceType_RR": car_telemetry["m_surfaceType"][3],
                    }
                    prediction = preprocess_and_predict([record], model, scaler)[0]
                    predictions.append(prediction)

                    # Aggiorna il grafico
                    line1.set_xdata(np.arange(len(records)))
                    line1.set_ydata([record["sessionTime"] for record in records])
                    line2.set_xdata(np.arange(len(predictions)))
                    line2.set_ydata(predictions)

                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    end_time = time.time()
                    print(f"Tempo di predizione: {end_time - start_time:.4f} secondi")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrotto dall'utente")
finally:
    if conn:
        conn.close()
    print("Chiusura del programma.")