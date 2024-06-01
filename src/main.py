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

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} ended. Logs: {logs}")


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


def test_train_model():
    file_path = "C:\\Users\\dadob\\Desktop\\F1_AI\\src\\data\\2023.json"

    with open(file_path, "r") as file:
        json_data = file.read()

    data = json.loads(json_data)

    # Normalizzare i dati JSON annidati
    header_df = pd.json_normalize(data["m_header"])
    telemetry_df = pd.json_normalize(data["m_carTelemetryData"])

    # Esplodere le colonne che contengono array
    brakes_temperature_df = telemetry_df["m_brakesTemperature"].apply(pd.Series)
    tyres_surface_temperature_df = telemetry_df["m_tyresSurfaceTemperature"].apply(pd.Series)
    tyres_inner_temperature_df = telemetry_df["m_tyresInnerTemperature"].apply(pd.Series)
    tyres_pressure_df = telemetry_df["m_tyresPressure"].apply(pd.Series)
    surface_type_df = telemetry_df["m_surfaceType"].apply(pd.Series)

    # Rinomina le colonne esplose
    brakes_temperature_df.columns = [
        f"m_brakesTemperature_{i}" for i in range(brakes_temperature_df.shape[1])
    ]
    tyres_surface_temperature_df.columns = [
        f"m_tyresSurfaceTemperature_{i}"
        for i in range(tyres_surface_temperature_df.shape[1])
    ]
    tyres_inner_temperature_df.columns = [
        f"m_tyresInnerTemperature_{i}"
        for i in range(tyres_inner_temperature_df.shape[1])
    ]
    tyres_pressure_df.columns = [
        f"m_tyresPressure_{i}" for i in range(tyres_pressure_df.shape[1])
    ]
    surface_type_df.columns = [
        f"m_surfaceType_{i}" for i in range(surface_type_df.shape[1])
    ]

    # Rimuovere le colonne originali dal DataFrame di telemetria
    telemetry_df.drop(
        columns=[
            "m_brakesTemperature",
            "m_tyresSurfaceTemperature",
            "m_tyresInnerTemperature",
            "m_tyresPressure",
            "m_surfaceType",
        ],
        inplace=True,
    )

    # Combinare i dati esplosi con i dati di telemetria
    telemetry_df = pd.concat(
        [
            telemetry_df,
            brakes_temperature_df,
            tyres_surface_temperature_df,
            tyres_inner_temperature_df,
            tyres_pressure_df,
            surface_type_df,
        ],
        axis=1,
    )

    # Combinare i dati header e telemetry
    df = pd.concat([header_df] * len(telemetry_df), ignore_index=True)
    df = pd.concat([df, telemetry_df], axis=1)

    # Visualizzare i nomi delle colonne
    print(df.columns)

    # Selezionare le colonne rilevanti
    target_columns = [
        "m_packetFormat",
        "m_gameYear",
        "m_gameMajorVersion",
        "m_gameMinorVersion",
        "m_packetVersion",
        "m_packetId",
        "m_sessionUID",
        "m_sessionTime",
        "m_frameIdentifier",
        "m_overallFrameIdentifier",
        "m_playerCarIndex",
        "m_secondaryPlayerCarIndex",
        "m_speed",
        "m_throttle",
        "m_steer",
        "m_brake",
        "m_clutch",
        "m_gear",
        "m_engineRPM",
        "m_drs",
        "m_revLightsPercent",
        "m_revLightsBitValue",
        "m_brakesTemperature_0",
        "m_brakesTemperature_1",
        "m_brakesTemperature_2",
        "m_brakesTemperature_3",
        "m_tyresSurfaceTemperature_0",
        "m_tyresSurfaceTemperature_1",
        "m_tyresSurfaceTemperature_2",
        "m_tyresSurfaceTemperature_3",
        "m_tyresInnerTemperature_0",
        "m_tyresInnerTemperature_1",
        "m_tyresInnerTemperature_2",
        "m_tyresInnerTemperature_3",
        "m_engineTemperature",
        "m_tyresPressure_0",
        "m_tyresPressure_1",
        "m_tyresPressure_2",
        "m_tyresPressure_3",
        "m_surfaceType_0",
        "m_surfaceType_1",
        "m_surfaceType_2",
        "m_surfaceType_3",
    ]

    # Verifica che tutte le colonne esistano
    missing_columns = [col for col in target_columns if col not in df.columns]
    print("Missing columns:", missing_columns)

    # Selezionare i dati per il training
    target = df[target_columns]

    # Espandi la dimensione del tuo input
    X_train = np.expand_dims(target.values, axis=1)

    # Convertire X_train in float32
    X_train = X_train.astype("float32")

    print(X_train.shape)

    # Selezionare la colonna target (ad esempio, 'm_speed')
    y_train = target["m_speed"].values

    # Reshape di y_train per avere la forma corretta
    y_train = y_train.reshape(-1, 1)

    # Convertire y_train in float32
    y_train = y_train.astype("float32")

    # Verificare le forme dei dati
    print("Shape di X_train:", X_train.shape)
    print("Shape di y_train:", y_train.shape)

    print("CPU disponibili:", tf.config.experimental.list_physical_devices("CPU"))
    print("GPU disponibili:", tf.config.experimental.list_physical_devices("GPU"))

    # Disabilita le GPU per evitare errori di memoria
    # print("Disabilitazione delle GPU...")
    # tf.config.experimental.set_visible_devices([], "GPU")

    # Definisci il modello sequenziale
    model = Sequential()

    # Aggiungi il livello di input, specificando la forma (1, 43)
    print("Aggiunta del livello di input...")
    model.add(Input(shape=(1, 43)))

    # Aggiungi un livello LSTM
    print("Aggiunta del livello LSTM...")
    model.add(LSTM(units=50, return_sequences=True))

    # Aggiungi un altro livello LSTM se necessario
    print("Aggiunta del secondo livello LSTM...")
    model.add(LSTM(units=50))

    # Aggiungi un livello denso per l'output
    print("Aggiunta del livello denso per l'output...")
    model.add(Dense(units=1))

    # Compila il modello
    print("Compilazione del modello...")
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    # Stampa il sommario del modello
    print("Sommario del modello:")
    model.summary()

    # Definisci i callback
    print("Definizione dei callback...")
    callbacks = [ProgbarLogger()]

    # Training del modello
    print("Training del modello...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callbacks)

    print("Fine del training.")
    print("Salvataggio del modello...")

    # Salva il modello
    model.save("f1_deep_learning_model.h5")
    joblib.dump(scaler, scaler_file_path)

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