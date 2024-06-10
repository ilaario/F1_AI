import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import fastf1


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

    def update(self, throttle, brake, clutch, gear, steer, drs, ers_mode):
        # Aggiorna la marcia
        if 0 < gear < len(self.gear_ratios):
            self.gear = gear

        # Calcola la nuova velocità e RPM in base all'accelerazione e alla marcia
        if throttle > 0:
            # Calcola l'effetto del DRS
            drs_effect = self.calculate_drs_effect(drs)

            # Calcola l'effetto dell'ERS
            ers_power = self.calculate_ers_power(ers_mode)

            # Calcola il tasso di accelerazione in base agli RPM
            base_acceleration_rate = 0.05 + 0.05 * (self.rpm / self.max_rpm)
            acceleration_rate = base_acceleration_rate + drs_effect + ers_power
            acceleration = throttle * acceleration_rate  # Acceleration rate
            self.speed += acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed

        if brake > 0:
            deceleration = brake * 34  # Deceleration rate
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

    def update_with_pos(self, throttle, brake, clutch, gear, x, y, drs, ers_mode):
        # Aggiorna la marcia
        if 0 < gear < len(self.gear_ratios):
            self.gear = gear

        # Calcola la nuova velocità e RPM in base all'accelerazione e alla marcia
        if throttle > 0:
            # Calcola l'effetto del DRS
            drs_effect = self.calculate_drs_effect(drs)

            # Calcola l'effetto dell'ERS
            ers_power = self.calculate_ers_power(ers_mode)

            # Calculate the proportional acceleration rate based on RPM using a logarithmic function
            max_acceleration_rate = 1.068
            acceleration_rate = max_acceleration_rate * (1 - np.log1p(self.rpm) / np.log1p(self.max_rpm))
            acceleration_rate = acceleration_rate + drs_effect + ers_power
            acceleration = throttle * acceleration_rate  # Acceleration rate
            self.speed += acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed

        if brake > 0:
            deceleration = brake * 25  # Deceleration rate
            self.speed -= deceleration
            if self.speed < 0:
                self.speed = 0

        # Calcola gli RPM in base alla velocità e alla marcia
        if self.gear > 0:
            self.rpm = self.speed * self.gear_ratios[self.gear]
            if self.rpm > self.max_rpm:
                self.rpm = self.max_rpm

        # Aggiorna la posizione della macchina
        self.position[0] = x
        self.position[1] = y

        return self.get_state()

    def calculate_drs_effect(self, drs):
        # Calcola l'effetto del DRS
        if drs in [2, 10, 12, 14]:
            return 0.01  # Incremento graduale di 10 km/h
        return 0

    def calculate_ers_power(self, ers_mode):
        # ERS power in kW
        max_power = 120  # Max power in kW
        ers_modes = {
            "medium": 0.03,
            "hotlap": 0.06,
            "overtake": 0.1
        }
        gear_modifier = {
            1: 0.0,
            2: ers_modes[ers_mode],
            3: ers_modes[ers_mode],
            4: ers_modes[ers_mode],
            5: ers_modes[ers_mode],
            6: ers_modes[ers_mode],
            7: ers_modes[ers_mode],
            8: ers_modes[ers_mode]
        }
        return gear_modifier[self.gear] * max_power / self.max_speed

    def set_start_state(self, speed, rpm, gear, position, direction):
        self.speed = speed
        self.rpm = rpm
        self.gear = gear
        self.position = position
        self.direction = direction

    def get_state(self):
        return {
            "speed": self.speed,
            "rpm": self.rpm,
            "gear": self.gear,
            "direction": self.direction,
            "position_x": self.position[0],
            "position_y": self.position[1]
        }


# Carica i dati di telemetria
session = fastf1.get_session(2023, 'Imola', 'Q')  # Esempio: Sessione di qualifica in Australia 2022
session.load()
ver = session.laps.pick_driver('VER').pick_fastest()
data = ver.get_car_data()  # Dati sulla frenata
speed_data = data['Speed']
rpm_data = data['RPM']
throttle_data = data['Throttle']
brake_data = data['Brake']
gear_data = data['nGear']
drs_data = data['DRS']

simulated_speed = []
simulated_rpm = []
simulated_gear = []
simulated_position = []


pos = ver.get_pos_data()  # Dati sulla posizione
x_data = pos['X']
y_data = pos['Y']

# Inizializza il simulatore
sim = CarSimulator()

sim.set_start_state(speed_data[0], rpm_data[0], gear_data[0] - 1, [x_data[0], y_data[0]], 0.0)
print("Start state:", sim.get_state())

for i in range(len(throttle_data)):
    sim.update_with_pos(throttle_data[i], brake_data[i], 0, gear_data[i] - 1, x_data[i], y_data[i], drs_data[i], "hotlap")
    print(sim.get_state())
    simulated_speed.append(sim.speed)
    simulated_rpm.append(sim.rpm)
    simulated_gear.append(sim.gear)
    simulated_position.append(sim.position)

# Plot della accelerazione simulata e reale
fig, ax = plt.subplots()
ax.plot(speed_data, label='Real Speed')
ax.plot(simulated_speed, label='Simulated Speed')
ax.set_xlabel('Time')
ax.set_ylabel('Speed')
ax.legend()
plt.show()

# Plot dei RPM simulati e reali
fig, ax = plt.subplots()
ax.plot(rpm_data, label='Real RPM')
ax.plot(simulated_rpm, label='Simulated RPM')
ax.set_xlabel('Time')
ax.set_ylabel('RPM')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(gear_data, label='Real Gear')
ax.plot([gear + 1 for gear in simulated_gear], label='Simulated Gear')
ax.set_xlabel('Time')
ax.set_ylabel('Gear')
ax.legend()
plt.show()
