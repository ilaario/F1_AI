import { F1TelemetryClient, constants } from 'f1-telemetry-client';
import net from 'net';

const { PACKETS } = constants;

// Configura il client TCP
const client = new F1TelemetryClient({ port: 20777 });
const TCP_PORT = 41234; // Porta dove lo script Python ascolterà
const TCP_HOST = '127.0.0.1'; // IP locale



const socket = new net.Socket();
socket.connect(TCP_PORT, TCP_HOST, () => {
  console.log('TLM: Connected to Python script');
});

console.log("TLM: Collecting data...");

client.on(PACKETS.carTelemetry, (data) => {
  const message = stringifyWithBigInt(data);
  //console.log(message);
  socket.write(message + '\n'); // Aggiungi newline per separare i messaggi
});

// to start listening:
client.start();
console.log("TLM: Client started");

// Funzione per serializzare BigInt
function stringifyWithBigInt(obj) {
    return JSON.stringify(obj, (key, value) =>
        typeof value === 'bigint' ? value.toString() + 'n' : value
    );
}
