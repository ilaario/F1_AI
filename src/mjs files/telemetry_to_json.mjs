import { F1TelemetryClient, constants } from 'f1-telemetry-client/src';
import { promises as fs } from 'fs';
import { stringify } from 'csv-stringify/sync';

const { PACKETS } = constants;

// Configura il client UDP
const client = new F1TelemetryClient({ port: 20777 });

const FILE_PATH = '/Users/ilaario/Desktop/AAU/Artificial Intelligence & Machine Learning/AI Project/src/data';

// scrive i dati su CSV
const telemetryData = {
    carTelemetry: [],
    carDamage: [],
    carSetups: [],
    carStatus: [],
    events: [],
    lapData: [],
    motion: [],
    tyreSets: []
};

// Funzione per scrivere i dati del buffer su disco
async function flushBufferToFile() {
    for (const key in telemetryData) {
        if (telemetryData[key].length > 0) {
            const data = telemetryData[key].map(item => JSON.parse(item));
            const csvData = stringify(data, { header: false });
            try {
                await fs.appendFile(`${FILE_PATH}/${key}.csv`, csvData, { flag: 'a' });
                console.log(`TLM: Data written to ${key}.csv`);
            } catch (err) {
                console.error(`Errore durante la scrittura del file ${key}.csv:`, err);
            }
            // Svuota il buffer dopo la scrittura
            telemetryData[key] = [];
        }
    }
}

// Esegui la funzione di scrittura periodica ogni 10 secondi
setInterval(flushBufferToFile, 5000);

client.on(PACKETS.carTelemetry, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.carTelemetry.push(message);
});

client.on(PACKETS.carDamage, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.carDamage.push(message);
});

client.on(PACKETS.carSetups, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.carSetups.push(message);
});

client.on(PACKETS.carStatus, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.carStatus.push(message);
});

client.on(PACKETS.event, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.events.push(message);
});

client.on(PACKETS.lapData, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.lapData.push(message);
});

client.on(PACKETS.motion, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.motion.push(message);
});

client.on(PACKETS.tyreSets, (data) => {
    const message = stringifyWithBigInt(data);
    console.log(message);
    telemetryData.tyreSets.push(message);
});

// to start listening:
client.start();
console.log("TLM: Client started");

// Funzione per convertire i BigInt in stringhe
function stringifyWithBigInt(obj) {
    return JSON.stringify(obj, (key, value) => {
        if (typeof value === 'bigint') {
            return value.toString();
        }
        return value;
    });
}