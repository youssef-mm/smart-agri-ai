from kafka import KafkaProducer
import pandas as pd
import json
import time
from pathlib import Path

TOPIC_NAME = "crop_yield_topic"
BATCH_SIZE = 1000

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "crop_yield.csv"

try:
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    chunks = pd.read_csv(DATA_FILE, chunksize=BATCH_SIZE)

    batch_number = 1

    for chunk in chunks:

        records = chunk.to_dict(orient="records")

        for record in records:
            producer.send(
                TOPIC_NAME,
                value=record
            )

        producer.flush()

        print(f"✅ Batch {batch_number} sent ({len(records)} records)")

        batch_number += 1

        time.sleep(0.5)

    producer.close()

    print("🎉 All batches sent successfully!")

except Exception as e:
    print(f"❌ Error: {e}")