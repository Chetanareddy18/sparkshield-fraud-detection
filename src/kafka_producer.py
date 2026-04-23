"""
Kafka Producer
==============
Streams the digital_payments dataset into a Kafka topic to simulate live
transactions for the SparkShield fraud detection pipeline.

Usage:
    # 1. start a local Kafka broker (e.g. via docker-compose) on localhost:9092
    # 2. python src/kafka_producer.py --topic transactions --rate 50

Each record is sent as JSON:
    {"step": 1, "type": "TRANSFER", "amount": 9839.64, ...}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "digital_payments.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="messages per second")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    args = parser.parse_args()

    try:
        from kafka import KafkaProducer
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "kafka-python not installed. Run: pip install kafka-python"
        ) from e

    print(f"Connecting to Kafka @ {args.bootstrap} ...")
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=50,
    )

    print(f"Reading {args.limit:,} rows from {args.data} ...")
    df = pd.read_csv(args.data, nrows=args.limit)

    delay = 1.0 / max(args.rate, 0.001)
    print(f"Streaming {len(df):,} messages to topic '{args.topic}' "
          f"at ~{args.rate} msg/s ...")

    sent = 0
    try:
        for row in df.to_dict(orient="records"):
            producer.send(args.topic, row)
            sent += 1
            if sent % 500 == 0:
                print(f"  sent {sent:,}")
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        producer.flush()
        producer.close()
        print(f"Done. Total sent: {sent:,}")


if __name__ == "__main__":
    main()
