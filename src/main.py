import os
import json
import pandas as pd
import joblib
import numpy as np
import socket
import matplotlib.pyplot as plt
import time
import subprocess
import tensorflow as tf
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ProgbarLogger, Callback
from sklearn.metrics import mean_squared_error
from io import StringIO
from threading import Thread, Event


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


def train_model(records):
    df = pd.DataFrame(records)
    df.dropna(inplace=True)

    features = df.drop(["sessionTime", "steer", "throttle", "brake"], axis=1)
    target = df[["steer", "throttle", "brake"]]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])))
    model.add(LSTM(50))
    model.add(Dense(3))  # 3 output: steer, throttle, brake

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
    )

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    model.save("f1_deep_learning_model.h5")
    joblib.dump(scaler, "scaler.pkl")

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
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Penalità per errori nelle sezioni critiche (es. velocità troppo alta)
    critical_threshold = 150.0  # Esempio di soglia critica per la velocità
    speed_penalty = tf.reduce_mean(
        tf.where(y_true[:, 0] > critical_threshold, tf.square(y_true[:, 0] - y_pred[:, 0]), 0.0))

    # Penalità per surriscaldamento gomme
    tyre_temp_threshold = 100.0  # Soglia di temperatura delle gomme (esempio)
    tyre_temp_penalty = tf.reduce_mean(
        tf.where(y_true[:, 20:24] > tyre_temp_threshold, tf.square(y_true[:, 20:24] - y_pred[:, 20:24]), 0.0))

    # Penalità per sterzata eccessiva
    steer_penalty = tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2]))

    return mse + speed_penalty + tyre_temp_penalty + steer_penalty

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

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/event_DRSE.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                event_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di events_DRSE.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/event_DRSD.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                event_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di events_DRSD.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/tyre_sets.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                tyre_sets_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di tyre_sets.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI '
              'Project/src/data/telemetry_data.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                car_telemetry_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di telemetry_data.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/car_setup.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                car_setups_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di car_setup.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/car_damage.json', 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                car_damage_data.append(record)
            except json.JSONDecodeError as e:
                print(f"Errore nella lettura di car_damage.json: {e}")
                continue

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/car_status.json', 'r') as file:
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

    with open('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/lap_data.json', 'r') as file:
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

def test_train_model():

    # Carica e normalizza i dati JSON
    df = load_and_normalize_json()

    # Verifica che tutte le colonne esistano
    print("Colonne disponibili nel DataFrame:", df.columns)

    # Selezionare solo le colonne numeriche per X_train
    X_train = df.select_dtypes(include=[np.number])

    # Rimuovere le colonne target da X_train se sono presenti
    y_columns = ['m_speed', 'm_throttle', 'm_steer', 'm_brake', 'm_clutch', 'm_gear', 'm_engineRPM',
                 'm_tyresSurfaceTemperature_0', 'm_tyresSurfaceTemperature_1', 'm_tyresSurfaceTemperature_2',
                 'm_tyresSurfaceTemperature_3']
    X_train = X_train.drop(columns=y_columns, errors='ignore')

    y_train = df[y_columns]

    # Espandi la dimensione del tuo input
    X_train = np.expand_dims(X_train.values, axis=1)

    # Convertire X_train e y_train in float32
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')

    # Verificare le forme dei dati
    print("Shape di X_train:", X_train.shape)
    print("Shape di y_train:", y_train.shape)

    # Carica lo scaler se esiste
    scaler_X_file_path = "scaler_X.pkl"
    scaler_y_file_path = "scaler_y.pkl"

    scaler_X = load_scaler(scaler_X_file_path)
    scaler_y = load_scaler(scaler_y_file_path)

    if scaler_X is None or scaler_y is None:
        # Se gli scaler non esistono, creane uno nuovo e salvalo
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        y_train_scaled = scaler_y.fit_transform(y_train)
        joblib.dump(scaler_X, scaler_X_file_path)
        joblib.dump(scaler_y, scaler_y_file_path)
    else:
        try:
            X_train_scaled = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            y_train_scaled = scaler_y.transform(y_train)
        except ValueError as e:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            y_train_scaled = scaler_y.fit_transform(y_train)
            joblib.dump(scaler_X, scaler_X_file_path)
            joblib.dump(scaler_y, scaler_y_file_path)

    # Definisci il modello sequenziale
    model = Sequential()

    print("Disabilitazione delle GPU...")
    tf.config.set_visible_devices([], 'GPU')

    print("Aggiunta del livello di input...")
    model.add(Input(shape=(1, X_train_scaled.shape[-1])))

    print("Aggiunta del livello LSTM...")
    model.add(LSTM(units=50, return_sequences=True))

    print("Aggiunta del secondo livello LSTM...")
    model.add(LSTM(units=50))

    print("Aggiunta del livello denso per l'output...")
    model.add(Dense(units=y_train.shape[-1]))

    print("Compilazione del modello con Adam e custom loss...")
    model.compile(optimizer='adam', loss=custom_loss)

    print("Sommario del modello:")
    model.summary()

    print("Training del modello...")
    callbacks = [ProgbarLogger(), CustomCallback()]
    model.fit(X_train_scaled, y_train_scaled, epochs=10, batch_size=32, callbacks=callbacks)

    print("Predizione con il modello addestrato...")
    y_pred_scaled = model.predict(X_train_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    mse = mean_squared_error(y_train, y_pred)
    print(f'Mean Squared Error: {mse}')

    model.save('f1_deep_learning_model.h5')
    joblib.dump(scaler_X, scaler_X_file_path)
    joblib.dump(scaler_y, scaler_y_file_path)

    print("Fine del programma.")


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
                print("Nessun dato ricevuto.")
                continue
            buffer += data
            try:
                packets = buffer.split("\n")
                buffer = packets.pop()
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


def start_node_client():
    global proc
    telemetry_script = "node src/telemetry.mjs"
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
    print("Training del modello con dati reali (1) o generati (2):")
    choice = input()
    if choice == "1":
        # Avvia il server TCP in un thread separato
        tcp_thread = Thread(target=start_tcp_server)
        tcp_thread.start()

        # Aspetta che il server TCP sia pronto
        server_ready.wait()

        # Avvia il client Node.js
        node_thread = Thread(target=start_node_client)
        node_thread.start()
        collect_data()
    elif choice == "2":
        test_train_model()
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