/*
 * AgroSense ML — ESP32 Smart Precision Agriculture Node Firmware
 *
 * Microcontroller: ESP32 DevKit v1
 * Sensors:
 *   - Analog Capacitive Soil Moisture Sensor v1.2 -> GPIO 34 (ADC1_CH6)
 *   - Digital DHT22 Temperature & Humidity Sensor -> GPIO 4
 * Actuators:
 *   - 5V Single Channel Relay Module (Active LOW) -> GPIO 23
 *
 * Cloud API: POST http://<SERVER_IP>:8000/predict
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h> // Library: ArduinoJson by Benoit Blanchon (v6 or v7)
#include <DHT.h>         // Library: DHT sensor library by Adafruit

// --- Network Configuration ---
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const char* API_ENDPOINT = "http://192.168.1.100:8000/predict";

// --- Hardware Pin Definitions ---
#define PIN_SOIL_ANALOG 34
#define PIN_DHT_DATA    4
#define PIN_RELAY_PUMP  23
#define DHT_TYPE        DHT22

// --- Soil Moisture Sensor Calibration Constants ---
// Calibrate by measuring raw ADC reading in air vs glass of water
const int AIR_ADC_VALUE   = 3200; // 0% moisture
const int WATER_ADC_VALUE = 1450; // 100% moisture

// --- Device Configuration ---
const char* DEVICE_ID = "esp32-field-node-01";
const char* CROP_TYPE = "Wheat";
const unsigned long TELEMETRY_INTERVAL_MS = 15000; // 15 seconds

DHT dht(PIN_DHT_DATA, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[AgroSense] Initializing ESP32 Smart Edge Node...");

  pinMode(PIN_RELAY_PUMP, OUTPUT);
  digitalWrite(PIN_RELAY_PUMP, HIGH); // Relay OFF (Active LOW)

  analogReadResolution(12); // 12-bit ADC (0 - 4095)
  dht.begin();

  connectToWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
  }

  // 1. Sample Physical Sensors
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  int rawSoilAdc = analogRead(PIN_SOIL_ANALOG);

  // Convert raw ADC to percentage volumetric moisture
  float soilMoisturePct = map(rawSoilAdc, AIR_ADC_VALUE, WATER_ADC_VALUE, 0, 100);
  soilMoisturePct = constrain(soilMoisturePct, 0.0, 100.0);

  // Fallback for DHT sensor read failure
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("[WARN] DHT reading failed, using baseline defaults");
    temperature = 25.0;
    humidity = 60.0;
  }

  Serial.printf("\n[Telemetry] Soil: %.1f%% | Temp: %.1f°C | Hum: %.1f%%\n",
                soilMoisturePct, temperature, humidity);

  // 2. Dispatch Telemetry to AgroSense ML API
  sendTelemetryAndActuate(soilMoisturePct, temperature, humidity);

  delay(TELEMETRY_INTERVAL_MS);
}

void connectToWiFi() {
  Serial.printf("[WiFi] Connecting to %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] Connection timeout. Operating in offline failsafe mode.");
  }
}

void sendTelemetryAndActuate(float soilMoisture, float temp, float humidity) {
  if (WiFi.status() != WL_CONNECTED) {
    applyOfflineFailsafe(soilMoisture);
    return;
  }

  HTTPClient http;
  http.begin(API_ENDPOINT);
  http.addHeader("Content-Type", "application/json");

  // Construct JSON Payload
  StaticJsonDocument<384> doc;
  doc["device_id"] = DEVICE_ID;
  doc["soil_moisture"] = round(soilMoisture * 10) / 10.0;
  doc["temperature"] = round(temp * 10) / 10.0;
  doc["humidity"] = round(humidity * 10) / 10.0;
  doc["rainfall_forecast_mm"] = 0.0; // In field: obtained from OpenWeatherMap or local rain gauge
  doc["soil_ph"] = 6.5;
  doc["nitrogen"] = 70.0;
  doc["phosphorus"] = 42.0;
  doc["potassium"] = 80.0;
  doc["crop_type"] = CROP_TYPE;
  doc["hour_of_day"] = 12;

  String requestBody;
  serializeJson(doc, requestBody);

  int httpCode = http.POST(requestBody);

  if (httpCode == 200) {
    String response = http.getString();
    StaticJsonDocument<512> respDoc;
    DeserializationError error = deserializeJson(respDoc, response);

    if (!error) {
      const char* action = respDoc["irrigation_action"];
      float waterLiters = respDoc["recommended_water_liters"];
      bool overrideActive = respDoc["override_active"];

      Serial.printf("[Cloud ML] Decision: %s | Water: %.2f L/m² | Override: %s\n",
                    action, waterLiters, overrideActive ? "ACTIVE" : "FALSE");

      if (strcmp(action, "ON") == 0 && !overrideActive) {
        digitalWrite(PIN_RELAY_PUMP, LOW);  // PUMP ON
        Serial.println("[Actuator] Pump Relay: ON");
      } else {
        digitalWrite(PIN_RELAY_PUMP, HIGH); // PUMP OFF
        Serial.println("[Actuator] Pump Relay: OFF");
      }
    } else {
      Serial.println("[ERR] JSON deserialize failed");
      applyOfflineFailsafe(soilMoisture);
    }
  } else {
    Serial.printf("[ERR] HTTP POST failed with code: %d\n", httpCode);
    applyOfflineFailsafe(soilMoisture);
  }

  http.end();
}

void applyOfflineFailsafe(float soilMoisture) {
  Serial.println("[Failsafe] Enforcing offline edge heuristics...");
  // Hard safety: never irrigate if soil is saturated (>75%)
  if (soilMoisture > 75.0) {
    digitalWrite(PIN_RELAY_PUMP, HIGH);
    Serial.println("[Failsafe] High moisture safeguard -> Pump forced OFF");
  } else if (soilMoisture < 22.0) {
    digitalWrite(PIN_RELAY_PUMP, LOW);
    Serial.println("[Failsafe] Critical drought threshold (<22%) -> Pump ON");
  } else {
    digitalWrite(PIN_RELAY_PUMP, HIGH);
    Serial.println("[Failsafe] Normal range -> Pump OFF");
  }
}
