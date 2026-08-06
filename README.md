# 🌱 Smart Agri AI

**Real-Time Big Data Pipeline for Smart Crop Yield Prediction**

A live, end-to-end data engineering and machine learning system that ingests, processes, and analyzes 1,000,000+ agricultural records (weather, soil, and sensor readings) in real time, and predicts crop yield instantly using streaming analytics and machine learning.

Final Project Proposal — 2026

---

## 📌 Overview

Smart Agri AI is a fully containerized, real-time pipeline built to solve a core problem in traditional agri-tech systems: **latency**. Instead of storing data first and analyzing it later, this system processes agricultural sensor data **the moment it arrives**, using in-memory streaming instead of slow disk-based storage.

The pipeline ingests live sensor data (soil moisture, soil ingredients, air humidity, temperature, UV) through Apache Kafka, processes and cleans it in-memory with PySpark Structured Streaming, feeds it into a trained MLlib regression model, and displays live predictions on an auto-refreshing Streamlit dashboard.

---

## 🚀 Key Features

- **Real-Time Ingestion** — Handles 1,000,000+ streaming records with zero bottlenecks via Apache Kafka.
- **In-Memory Processing** — No slow disk I/O; PySpark Streaming processes data entirely in memory for maximum speed.
- **Accurate Yield Prediction** — Predicts crop yield (tons/hectare) as soon as new readings arrive, using a Random Forest Regressor.
- **One-Command Deployment** — The entire stack (Kafka, Zookeeper, Spark) runs via a single `docker-compose up` command.
- **Live Dashboard** — An auto-refreshing Streamlit UI visualizes streaming results and supports live manual inference.
- **Orchestrated Pipeline** — Apache Airflow DAGs manage and schedule the end-to-end workflow.

---

## 🏗️ Architecture

```
Sensors / Data Source
        │
        ▼
  Apache Kafka (Streaming Buffer)
        │
        ▼
  PySpark Structured Streaming (In-Memory ETL)
        │
        ▼
  PySpark MLlib (Random Forest Regressor)
        │
        ▼
  Streamlit Dashboard (Live Inference + Visualization)
```

**Pipeline stages:**

1. **Apache Kafka** — High-throughput streaming buffer that receives sensor/weather data as it's generated, without ever blocking the producer.
2. **PySpark Structured Streaming** — Consumes data from Kafka, cleans it, handles missing values, and runs Spark SQL analytics — all in memory, with no intermediate storage.
3. **PySpark MLlib** — Feature engineering (StringIndexer, VectorAssembler), model training (Random Forest Regressor), and evaluation (RMSE, R²).
4. **Streamlit Dashboard** — Auto-refreshing UI that displays live streaming results and lets users input field data for instant yield predictions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Streaming / Messaging | Apache Kafka |
| Data Processing | PySpark (Structured Streaming, Spark SQL) |
| Machine Learning | PySpark MLlib (Random Forest Regressor) |
| Orchestration | Apache Airflow |
| Dashboard / UI | Streamlit |
| Containerization | Docker & Docker Compose |
| Language | Python |

### Why these technologies?

- **Apache Kafka** — Acts as a high-speed streaming buffer capable of absorbing millions of readings per second without the system going down.
- **PySpark Streaming** — Processes a million records in-memory, dramatically faster than traditional tools like Pandas at scale.
- **Docker** — Unifies the entire runtime environment so the project runs identically on any machine with one command.
- **Streamlit** — 100% Python, integrates seamlessly with the ML models, and supports automatic UI refresh for live results.

---

## 📁 Project Structure

```
smart-agri-ai/
├── dags/                          # Airflow DAGs for pipeline orchestration
│   ├── agri_pipeline_dag.py
│   └── data/
│       └── crop_yield.csv
│
├── docker/                        # Containerization setup
│   └── docker-compose.yml
│
├── models/                        # Trained ML models
│   ├── rf_crop_yield_model/
│   └── rf_crop_yield_spark_model/
│       └── rf_crop_yield_model.joblib
│
├── src/
│   ├── ingestion/                 # Kafka producer for live data streaming
│   │   └── kafka_producer.py
│   │
│   ├── spark/                     # PySpark ETL & analytics
│   │   ├── preprocessing.py
│   │   └── analytics_sql.py
│   │
│   ├── ml/                        # Model training & evaluation
│   │   ├── train_mllib.py
│   │   └── evaluate_mllib.py
│   │
│   └── dashboard/                 # Streamlit application
│       ├── app.py
│       ├── data_loader.py
│       ├── generator.py
│       ├── predictor.py
│       ├── styling.py
│       ├── utils.py
│       └── visuals.py
│
├── .streamlit/                    # Streamlit configuration
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Apache Airflow (if running the orchestration layer locally)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/youssef-mm/smart-agri-ai.git
   cd smart-agri-ai
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Spin up the infrastructure (Kafka, Zookeeper, Spark)**
   ```bash
   cd docker
   docker compose up -d
   ```

4. **Start streaming data into Kafka**
   ```bash
   python src/ingestion/kafka_producer.py
   ```

5. **Run the PySpark streaming & analytics job**
   ```bash
   python src/spark/preprocessing.py
   ```

6. **Train / evaluate the ML model** *(optional — pre-trained models are included in `/models`)*
   ```bash
   python src/ml/train_mllib.py
   python src/ml/evaluate_mllib.py
   ```

7. **Launch the dashboard**
   ```bash
   streamlit run src/dashboard/app.py
   ```

---

## 📊 Model Performance

The Random Forest Regressor is evaluated using:

- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

Trained models are exported to the `/models` directory for reuse without retraining.

---

## 🌿 Git Workflow

To keep collaboration clean and error-free, the team follows strict conventions:

- **`.gitignore`** — Large datasets and model checkpoints are excluded from version control.
- **Feature Branches** — Every team member works exclusively on their own dedicated feature branch.
- **Protected `main`** — Direct pushes to `main` are disabled; all changes go through Pull Requests and code review.

---

## 👥 Team & Responsibilities

| Member | Main Focus | Branch | Output |
|---|---|---|---|
| **Jana Atef** | Infrastructure & Streaming | `feature/docker-kafka-ingestion` | Docker Compose & Kafka Producer |
| **Mohamed Emad** | Big Data Engineer | `feature/spark-streaming-sql` | PySpark Streaming & Spark SQL |
| **Rowaida Mohamed** | Machine Learning Engineer | `feature/mllib-crop-prediction` | MLlib Crop Prediction (RMSE, R²) |
| **Youssef Mohamed** | Visualization & Team Lead | `feature/streamlit-visualization` | Streamlit Dashboard & Live UI |

---

## 📄 License

This project was developed as a final academic project (NTI Big Data track, 2026).

---

## 📬 Contact

For questions or feedback, feel free to open an issue on this repository.
## Thank You

