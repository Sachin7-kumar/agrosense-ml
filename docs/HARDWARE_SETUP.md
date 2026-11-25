# AgroSense ML — IoT Edge Hardware & Circuit Wiring Guide

This guide details the physical hardware architecture, circuit schematic, pinout assignments, and sensor calibration procedures for deploying the AgroSense ML smart irrigation edge node.

---

## 1. Bill of Materials (BOM)

| Component | Specification | Operating Voltage | Purpose | Approx. Cost (USD) |
| :--- | :--- | :--- | :--- | :--- |
| **Microcontroller** | ESP32 DevKit v1 (30-pin, CP2102) | 3.3V / 5V (USB) | Edge telemetry, Wi-Fi client, relay actuation | \$4.50 |
| **Soil Moisture Sensor** | Capacitive Soil Moisture Sensor v1.2 | 3.3V | Corrosion-resistant analog volumetric soil moisture | \$1.80 |
| **Temp / Humidity Sensor** | DHT22 (AM2302) or SHT31 | 3.3V - 5V | Digital ambient temperature and relative humidity | \$3.50 |
| **Actuator Relay** | 5V Single-Channel Relay Module (Optocoupled, Active LOW) | 5V (VCC), 3.3V logic | Switches high-current water pump / solenoid valve | \$1.20 |
| **Water Pump / Valve** | 12V DC Submersible Mini Pump or 12V Solenoid Valve | 12V DC | Physical irrigation delivery | \$6.00 |
| **Power Supply** | 12V 2A DC Adapter + LM2596 Buck Converter (to 5V) | 12V -> 5V | Supplies pump (12V) and ESP32 DevKit (5V Vin) | \$5.00 |
| **Passives & Misc** | 10kΩ pull-up resistor, breadboard / PCB, jumper wires | — | Circuit assembly | \$2.00 |
| **Total** | | | | **~\$24.00** |

---

## 2. Circuit Wiring & Pinout Table

```text
       +--------------------------------------------------+
       |               ESP32 DevKit v1 (30-Pin)           |
       |                                                  |
       |   3V3  ----------------- VCC (Capacitive Soil)   |
       |   GND  ----------------- GND (Capacitive Soil)   |
       |   GPIO 34 (ADC1_CH6) --- AOUT (Capacitive Soil)  |
       |                                                  |
       |   3V3  ----------------- VCC (DHT22)             |
       |   GND  ----------------- GND (DHT22)             |
       |   GPIO 4  -------------- DATA (DHT22 + 10k Pullup)|
       |                                                  |
       |   VIN (5V) ------------- VCC (5V Relay Module)   |
       |   GND  ----------------- GND (5V Relay Module)   |
       |   GPIO 23 -------------- IN (5V Relay Trigger)   |
       +--------------------------------------------------+
                                  |
                           Relay COM / NO
                                  |
              [+12V Power] ---> [Relay] ---> [12V Water Pump] ---> [GND]
```

### Detailed Pin Connections

1. **Capacitive Soil Moisture Sensor v1.2**:
   - `VCC` -> ESP32 `3V3`
   - `GND` -> ESP32 `GND`
   - `AOUT` -> ESP32 `GPIO 34` (ADC1 channel; safe to use with active Wi-Fi).

2. **DHT22 Temperature & Humidity**:
   - `Pin 1 (VCC)` -> ESP32 `3V3`
   - `Pin 2 (DATA)` -> ESP32 `GPIO 4` (with a 10kΩ pull-up resistor between Pin 1 and Pin 2).
   - `Pin 4 (GND)` -> ESP32 `GND`

3. **5V Optocoupled Relay Module**:
   - `VCC` -> ESP32 `VIN` (5V rail)
   - `GND` -> ESP32 `GND`
   - `IN` -> ESP32 `GPIO 23`
   - **Load Side**: Connect `COM` to +12V DC power supply, `NO` (Normally Open) to the positive terminal of the 12V Solenoid Valve / Pump, and return pump negative terminal to 12V GND.

---

## 3. Sensor Calibration Procedure

Analog capacitive soil moisture probes measure the dielectric permittivity of the surrounding medium. To map raw 12-bit ADC values ($0 - 4095$) to accurate volumetric moisture percentages ($0\% - 100\%$):

1. **Air Calibration (0% Dry Baseline)**:
   - Hold the clean probe dry in the air.
   - Record the raw ADC value: typically `AIR_ADC_VALUE ≈ 3100 - 3300`.
2. **Water Calibration (100% Saturation Baseline)**:
   - Submerge the probe blade in tap water up to the white indicator line (do NOT submerge circuit components).
   - Record the raw ADC value: typically `WATER_ADC_VALUE ≈ 1400 - 1550`.
3. **Firmware Mapping Function**:
   ```cpp
   float soilMoisturePct = map(rawSoilAdc, AIR_ADC_VALUE, WATER_ADC_VALUE, 0, 100);
   soilMoisturePct = constrain(soilMoisturePct, 0.0, 100.0);
   ```

---

## 4. Hardware Fail-Safe Mechanism

In real agricultural settings, Wi-Fi or cellular gateways can experience intermittency. The ESP32 edge firmware (`scripts/esp32_firmware_sample.ino`) implements hard local failsafes:
- **Lockout**: If soil moisture $> 75\%$ or rainfall is detected, the relay pin is forced `HIGH` (Relay OFF), preventing crop root rot or flood damage regardless of network state.
- **Drought Survival**: If network connectivity drops and soil moisture drops below critical wilting threshold ($< 22\%$), the local controller triggers an emergency 5-minute hydration cycle.
