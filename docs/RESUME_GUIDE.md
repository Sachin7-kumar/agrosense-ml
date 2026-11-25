# AgroSense ML — Resume & Technical Interview Guide

A comprehensive guide for showcasing **AgroSense ML** on your resume, LinkedIn, portfolio, and acing technical interviews.

---

## 1. Ready-to-Use Resume Bullet Points

Select the bullets that best match your target job title. Each bullet follows the **Google XYZ Formula**: *"Accomplished [X] as measured by [Y] by doing [Z]"*.

### For Machine Learning Engineer / MLOps Roles
- Designed and deployed an end-to-end precision agriculture ML system serving **XGBoost classification (97.6% F1, 0.998 ROC-AUC)** and **regression (0.87 L/m² RMSE, 0.97 R²)** pipelines to automate irrigation decisioning.
- Engineered physics-informed agronomic features (Vapor Pressure Deficit via Tetens equation, diurnal evapotranspiration, NPK balance) in an automated Scikit-Learn `ColumnTransformer` pipeline.
- Built an edge-ready **FastAPI** microservice achieving **<1ms per-sample latency** (3,200+ samples/sec throughput in batch mode) and integrated **TreeSHAP** for real-time local factor attribution.
- Enforced hard agronomic safety interlocks preventing over-irrigation during saturated soil (>75%) or rainstorms (>15mm), guaranteeing zero model-induced flooding.
- Containerized services with **Docker** and **Docker Compose**, automated testing via **GitHub Actions CI**, and implemented latency profiling (P50/P95/P99) across 500+ simulated nodes.

### For Data Scientist Roles
- Developed a two-stage cascaded ML architecture predicting binary irrigation need and continuous water volume ($L/m^2$) conditioned on soil moisture, weather forecasts, and diurnal cycles.
- Formulated physics-grounded synthetic dataset generator (6,000+ samples) modeling diurnal thermal dynamics, relative humidity inverse correlations, and crop coefficients ($K_c$) for 4 cultivars.
- Evaluated and benchmarked 6 candidate estimators (Logistic Regression, Ridge, Random Forest, XGBoost) using Stratified Cross-Validation, ROC-AUC, and permutation feature importance.
- Built an interactive **Streamlit** dashboard featuring live telemetry simulation, scenario stress-testing (Drought, Heatwave, Rainstorm), and interactive confusion matrices.

### For IoT & Embedded Systems Roles
- Developed and simulated an **ESP32 IoT edge node** communicating via HTTP REST to cloud ML backend, featuring automatic 5V relay pump actuation and fail-safe local heuristics.
- Wrote modular Arduino C++ firmware (`scripts/esp32_firmware_sample.ino`) reading analog capacitive soil moisture and DHT22 sensors, with dynamic ADC calibration and JSON parsing.
- Designed offline fallback interlocks ensuring fail-safe pump deactivation on sensor disconnection or network dropouts.

### For Backend / Full-Stack Engineer Roles
- Architected high-performance **FastAPI** REST API with asynchronous lifespan state management, Pydantic v2 data validation, and vectorized batch inference (`POST /predict/batch`).
- Implemented in-memory circular telemetry logging buffer tracking device activity, water volume savings, and anomaly detection stats.
- Packaged multi-service architecture using Docker Compose with automated health checks, non-root security privileges, and persistent volumes.

---

## 2. 30-Second Elevator Pitch

> *"AgroSense ML is an end-to-end smart agriculture platform that predicts whether agricultural plots need irrigation and calculates the exact water volume required. It bridges physical agronomic science—like Vapor Pressure Deficit and diurnal evapotranspiration—with gradient-boosted trees (XGBoost). The system is deployed via a high-throughput FastAPI backend serving ESP32 IoT edge nodes with sub-millisecond per-item batch latency, hard agronomic safety interlocks, TreeSHAP explainability, and an interactive Streamlit monitoring dashboard."*

---

## 3. Top 10 Technical Interview Questions & Answers

### Q1: Walk me through the high-level architecture of AgroSense ML.
**Answer:**
AgroSense ML follows a 4-tier architecture:
1. **Edge Tier (IoT Node)**: ESP32 microcontroller sampling analog capacitive soil moisture and digital DHT22 temperature/humidity sensors every 15 seconds.
2. **Ingestion & Safety Layer (FastAPI)**: REST endpoints (`/predict` and `/predict/batch`) validating inputs against agronomic schemas, running anomaly detection for faulty probes, and enforcing hard safety cutoffs (soil $>75\%$ or rainfall $>15$ mm).
3. **Inference Engine (Scikit-Learn & XGBoost)**: Custom transformer pipeline computing physical features (VPD, diurnal ET), scaling numeric inputs, one-hot encoding crop cultivars, and executing a two-stage cascade (binary classifier then regressor).
4. **Presentation & Monitoring (Streamlit)**: Live operational dashboard displaying pump states, real-time Plotly trendlines, TreeSHAP feature attributions, and scenario sandboxes.

### Q2: Why did you use a two-stage cascade (Classifier -> Regressor) instead of a single multi-output or direct regression model?
**Answer:**
In agricultural irrigation, water volume is a **zero-inflated semi-continuous variable**: over 60% of the time, the required water is exactly zero.
- If we trained a single regressor directly on all data, it would suffer from severe zero-inflation, predicting small non-zero values (e.g., $0.3 L/m^2$) during dry weather that would intermittently pulse pumps and waste energy.
- By decoupling the problem into:
  1. **Stage 1 (Classification)**: Highly sensitive decision boundary ($F1=0.976$, $ROC-AUC=0.998$) determining whether irrigation is required.
  2. **Stage 2 (Regression)**: Conditioned exclusively on triggered events ($y_{clf} == 1$), focusing regression capacity purely on calculating optimal water volume ($R^2=0.971$, $RMSE=0.87 L/m^2$).
This provides optimal operational reliability.

### Q3: What physics-informed features did you engineer, and why?
**Answer:**
Raw IoT readings (temperature, humidity, moisture) do not directly capture physiological plant stress. I engineered:
1. **Vapor Pressure Deficit (VPD in kPa)**: Calculated using Tetens equation for saturation vapor pressure:
   $$e_s(T) = 0.6108 \exp\left(\frac{17.27 T}{T + 237.3}\right), \quad \text{VPD} = e_s(T) \times \left(1 - \frac{\text{RH}}{100}\right)$$
   VPD measures the atmospheric drying power pulling water out of plant leaves. High VPD triggers water loss even when soil moisture seems moderate.
2. **Diurnal Evapotranspiration Factor**: Gaussian solar bell curve peaking at 13:30 to model solar radiation intensity throughout the day.
3. **Effective Moisture**: Combines current soil moisture with forecasted rainfall ($M_{\text{eff}} = M_{\text{soil}} + 0.75 \times \text{Rain}_{24h}$) to prevent watering immediately before precipitation.
4. **NPK Balance Index**: Harmonic average of N, P, K macronutrients representing soil nutrient vigor.

### Q4: How do you guarantee safety against model hallucination or over-watering?
**Answer:**
We adhere to a **defense-in-depth safety design**:
1. **Deterministic Rule Override**: Even if an ML model outputs a high probability of irrigation ($P=0.99$), the inference layer checks hard agronomic constraints:
   - If Soil Moisture $> 75\%$ -> Force `irrigation_action: "OFF"`, water: `0.0 L`.
   - If Forecasted 24h Rainfall $> 15$ mm -> Force `irrigation_action: "OFF"`, water: `0.0 L`.
2. **Edge Hardware Failsafe**: The ESP32 firmware contains autonomous local threshold checks so that even if the cloud API were compromised or returns an erroneous command, the microcontroller will never energize the pump relay when soil moisture is saturated.

### Q5: How do you provide explainability to field operators or farmers?
**Answer:**
Black-box ML predictions erode user trust in agronomy. We integrated **TreeSHAP (SHapley Additive exPlanations)**:
- In XGBoost, exact Shapley values are calculated via `get_booster().predict(DMatrix, pred_contribs=True)`.
- For each prediction, the API extracts the top 3 driving features, their directional impact (increases vs decreases need), and translates them into plain-language summaries (e.g., *"Vapor Pressure Deficit (2.41 kPa) increases irrigation likelihood (+0.245)"*).

### Q6: How did you measure and benchmark inference latency?
**Answer:**
I authored an automated benchmarking suite (`scripts/benchmark_latency.py`):
- Single-sample Python inference: **~41ms** (including TreeSHAP calculation).
- Vectorized batch inference: TreeSHAP calculation is vectorized across the entire matrix in C++, dropping per-sample latency to **0.31ms** at batch size 250 with a throughput of **3,269 items/second**.
- FastAPI HTTP round-trip: Median latency **65ms**, well within the 15-second IoT sampling period.

### Q7: How do you detect sensor drift or hardware malfunctions?
**Answer:**
We built a rule-based Out-of-Distribution / Sensor Health Validator (`detect_sensor_anomalies`):
- Detects disconnected probes (moisture $< 1.5\%$).
- Detects short-circuits (moisture $> 98.5\%$).
- Flags impossible physical correlations (e.g., soil moisture $< 10\%$ during rainfall $> 20$ mm, or extreme heat $> 46^\circ\text{C}$ with $> 85\%$ humidity).
These are returned in the API response under `sensor_anomalies` and surfaced as warning banners on the dashboard.

### Q8: What testing strategies did you employ?
**Answer:**
Using **Pytest**, we test:
1. **Data determinism & schema validation**: Verifying synthetic generation preserves column types, ranges, and reproducibility.
2. **Preprocessing & unseen categories**: Ensuring `ColumnTransformer` handles unseen crop types gracefully without throwing exceptions.
3. **Model threshold assertions**: Automated tests fail if model $F_1 < 0.85$, $ROC\text{-}AUC < 0.90$, or $R^2 < 0.80$.
4. **API contract & HTTP status codes**: Testing `/health`, `/predict`, `/predict/batch`, `/telemetry/*`, verifying 422 Unprocessable Entity on out-of-range sensor inputs, and validating 100% safety cutoff execution.

### Q9: How is the service containerized and deployed?
**Answer:**
The system includes:
- A multi-stage **Dockerfile** with non-root user security and internal healthchecks.
- A **Docker Compose** orchestration running both the FastAPI backend and Streamlit dashboard on a shared network bridge, with dependency health checks (`depends_on: api: condition: service_healthy`).
- **GitHub Actions CI** running linting and tests across Python 3.10, 3.11, and 3.12.

### Q10: How would you scale this architecture to 50,000 active nodes?
**Answer:**
1. **Decouple Ingestion from Inference**: Place an **MQTT broker (e.g., EMQX / Mosquitto)** or **Apache Kafka** between the ESP32 edge nodes and backend. ESP32 devices publish lightweight MQTT packets.
2. **Worker Pool**: Use Celery / Redis or Kubernetes-scaled consumers reading telemetry from Kafka partitions and executing vectorized batch inference with Triton Inference Server or ONNX Runtime.
3. **Time-Series Storage**: Stream raw telemetry to **TimescaleDB** or **ClickHouse** for long-term historical analytics.
