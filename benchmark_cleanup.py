import os
import time
import datetime
from peewee import *
from playhouse.sqlite_ext import *
from dataclasses import dataclass
from typing import Any

# create mock db
db = SqliteDatabase(':memory:')

class Event(Model):
    id = CharField(primary_key=True)
    label = CharField()
    camera = CharField()
    start_time = FloatField()
    end_time = FloatField()
    thumbnail = TextField()
    has_clip = BooleanField(default=True)
    has_snapshot = BooleanField(default=True)
    retain_indefinitely = BooleanField(default=False)
    data = JSONField()

    class Meta:
        database = db

db.connect()
db.create_tables([Event])

# insert mock data
print("Inserting mock data...")
labels = ['person', 'car', 'cat', 'dog', 'bicycle', 'motorcycle', 'bird', 'horse']
cameras = ['front', 'back', 'removed_cam']

events_to_insert = []
now = datetime.datetime.now().timestamp()
for i in range(10000):
    label = labels[i % len(labels)]
    camera = cameras[i % len(cameras)]
    # make some older than 10 days, some older than 5 days, some recent
    start_time = now - (i % 20) * 86400

    events_to_insert.append({
        'id': f"event_{i}",
        'label': label,
        'camera': camera,
        'start_time': start_time,
        'end_time': start_time + 10,
        'thumbnail': '',
        'has_clip': True,
        'has_snapshot': True,
        'retain_indefinitely': False,
        'data': {"max_severity": "detection"},
    })

with db.atomic():
    Event.insert_many(events_to_insert).execute()

print(f"Inserted {Event.select().count()} events")

@dataclass
class MockRetainObject:
    default: int
    objects: dict[str, int]

@dataclass
class MockRetainConfig:
    retain: MockRetainObject

@dataclass
class MockCameraConfig:
    snapshots: MockRetainConfig

class MockConfig:
    def __init__(self):
        self.cameras = {
            'front': MockCameraConfig(MockRetainConfig(MockRetainObject(10, {'person': 15, 'car': 5}))),
            'back': MockCameraConfig(MockRetainConfig(MockRetainObject(10, {'cat': 2}))),
        }

config = MockConfig()

@dataclass
class DistinctLabel:
    label: str

def baseline():
    events_to_update = []

    # Simulate expire_snapshots behavior
    for name, camera in config.cameras.items():
        retain_config = camera.snapshots.retain
        distinct_labels = [DistinctLabel(l) for l in labels]

        for event in distinct_labels:
            expire_days = retain_config.objects.get(
                str(event.label), retain_config.default
            )

            expire_after = (
                datetime.datetime.now() - datetime.timedelta(days=expire_days)
            ).timestamp()

            expired_events = (
                Event.select(
                    Event.id,
                    Event.camera,
                    Event.thumbnail,
                )
                .where(
                    Event.camera == name,
                    Event.start_time < expire_after,
                    Event.label == event.label,
                    Event.retain_indefinitely == False,
                )
                .namedtuples()
                .iterator()
            )

            for ev in expired_events:
                events_to_update.append(str(ev.id))

    return len(events_to_update)

def optimized():
    events_to_update = []

    # Simulate expire_snapshots behavior with optimization
    for name, camera in config.cameras.items():
        retain_config = camera.snapshots.retain
        distinct_labels = [DistinctLabel(l) for l in labels]

        # We need to map labels to their respective expire_days
        # Group labels by expire_days to minimize queries
        expire_days_to_labels = {}
        for event in distinct_labels:
            expire_days = retain_config.objects.get(
                str(event.label), retain_config.default
            )
            if expire_days not in expire_days_to_labels:
                expire_days_to_labels[expire_days] = []
            expire_days_to_labels[expire_days].append(str(event.label))

        for expire_days, grouped_labels in expire_days_to_labels.items():
            expire_after = (
                datetime.datetime.now() - datetime.timedelta(days=expire_days)
            ).timestamp()

            expired_events = (
                Event.select(
                    Event.id,
                    Event.camera,
                    Event.thumbnail,
                )
                .where(
                    Event.camera == name,
                    Event.start_time < expire_after,
                    Event.label << grouped_labels,
                    Event.retain_indefinitely == False,
                )
                .namedtuples()
                .iterator()
            )

            for ev in expired_events:
                events_to_update.append(str(ev.id))

    return len(events_to_update)


# run benchmarks
n_runs = 50

print("Running baseline...")
start_time = time.time()
for _ in range(n_runs):
    baseline()
baseline_time = time.time() - start_time
print(f"Baseline: {baseline_time:.4f} seconds")

print("Running optimized...")
start_time = time.time()
for _ in range(n_runs):
    optimized()
optimized_time = time.time() - start_time
print(f"Optimized: {optimized_time:.4f} seconds")

print(f"Improvement: {(baseline_time - optimized_time) / baseline_time * 100:.2f}%")
