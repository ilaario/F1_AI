import socket
import json
import time

def load_packets_from_file(file_path):
    with open(file_path, 'r') as file:
        packets = json.load(file)
    return packets

def start_client(host='127.0.0.1', port=41234):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Connesso al server {host}:{port}")


    try:
        start_time = time.time()
        while time.time() - start_time < 9:
            carDamage_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/car_damage.json')
            json_data = json.dumps(carDamage_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'carDamage' inviato")
            time.sleep(1)  # Attendi un secondo tra l'invio dei pacchetti

            carSetup_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/car_setup.json')
            json_data = json.dumps(carSetup_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'carSetup' inviato")
            time.sleep(1)

            carTelemetry_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/telemetry_data.json')
            json_data = json.dumps(carTelemetry_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'carTelemetry' inviato")
            time.sleep(1)

            carStatus_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/car_status.json')
            json_data = json.dumps(carStatus_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'carStatus' inviato")
            time.sleep(1)

            event_packet = load_packets_from_file('/src/data/events.json')
            json_data = json.dumps(event_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'event_DRSD' inviato")
            time.sleep(1)

            event_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/event_DRSE.json')
            json_data = json.dumps(event_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'event_DRSE' inviato")
            time.sleep(1)

            lapData_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/lap_data.json')
            json_data = json.dumps(lapData_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'lapData' inviato")
            time.sleep(1)

            motion_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/motion.json')
            json_data = json.dumps(motion_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'motion' inviato")
            time.sleep(1)

            tyreSet_packet = load_packets_from_file('/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data/tyre_sets.json')
            json_data = json.dumps(tyreSet_packet)
            client_socket.sendall(json_data.encode('utf-8') + b'\n')
            print("Pacchetto 'tyreSet' inviato")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrotto dall'utente")
    finally:
        client_socket.close()
        print("Connessione chiusa")

if __name__ == "__main__":
    start_client()