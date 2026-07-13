# Frigate NVR — Code Review Issues

> **Date:** 2026-06-10
> **Hardware Context:** AMD Ryzen 5 3500U / Radeon Vega 8 Mobile (Picasso) / 14 GB RAM / NVMe SSD
> **Codebase:** `/home/manu/Documents/github-orgs/Zektopic/frigate`

---

## Critical Issues

### 1. Unsafe JWT Secret Key Derivation
**Files:** `frigate/api/fastapi_app.py:167-186`, `frigate/api/auth.py:307-361`

The JWT secret from `get_jwt_secret()` returns a string, then `fastapi_app.py` conditionally tries to decode it as hex bytes. If the hex check fails (e.g., a secret from Home Assistant options containing non-hex chars), the raw UTF-8 bytes are used directly — weaker than the original hex entropy.

The `len(jwt_secret) < 64` warning at `auth.py:358` is advisory only and does not reject weak secrets. A secret shorter than 64 bytes produces a weaker HMAC key.

**Impact:** Reduced JWT security if non-hex secrets are used. Attackers may brute-force shorter secrets more easily.

### 2. JWT Not Invalidated on Every Request After Password Change
**File:** `frigate/api/auth.py:720-739`

When a user changes their password, existing JWTs are only checked during cookie **refresh** — not on every authenticated request. An attacker with a stolen JWT can continue using it until the refresh window expires. The `iat` vs `password_changed_at` comparison only happens inside the refresh block.

**Impact:** Stolen tokens remain valid longer than intended after password rotation.

### 3. Race Condition: Stale Detection Results in Shared Memory
**File:** `frigate/object_detection/base.py:415-425`

`RemoteObjectDetector.detect()` drains stale ZMQ messages before sending a new detection request, but does NOT clear `self.out_np_shm`. If the old detection already wrote results to shared memory before the stale message was drained, the new detection could read stale data from `out_np_shm`. This is a race between the ZMQ drain and the SHM overwrite.

**Impact:** False detections or missed objects under high detection load.

---

## High-Severity Issues

### 4. Silent F-String Expressions — No Logging on Error
**File:** `frigate/comms/dispatcher.py`

Three locations have bare f-strings where `logger.error()` was clearly intended:

- **Line 594:** `f"Received unsupported value for motion contour area: {payload}"` — string created and discarded
- **Lines 608-609:** Same bug in `_on_motion_threshold_command`
- **Lines 623-624:** Same bug in `_on_global_notification_command`

**Impact:** Invalid MQTT commands are silently ignored with zero observability. Operators have no way to know their automation commands are failing.

### 5. Unsafe Pickle Serialization via SyncManager
**Files:** `frigate/__main__.py:19`, `frigate/app.py`

The app uses `mp.Manager()` and `manager.Queue()` extensively. `SyncManager` serializes all objects with `pickle`, which can execute arbitrary code during deserialization. Data flowing through MQTT and WebSocket paths eventually enters these queues.

**Impact:** If an attacker can inject data through MQTT/WS (e.g., unauthenticated on LAN), they could potentially achieve remote code execution.

### 6. Dual Database Connection Pools on Same File
**File:** `frigate/record/record.py:29-41` vs `frigate/app.py:261`

The main process binds a `SqliteVecQueueDatabase` with `synchronous=NORMAL`, while `RecordProcess` creates its own `SqliteQueueDatabase` with the same pragma. While SQLite WAL mode allows concurrent access, two independent connection pools with potentially diverging pragma settings writing to the same file is fragile. If one pool changes connection settings (e.g., cache size), the other is unaware.

**Impact:** Potential database corruption under heavy concurrent write load.

### 7. Storage Cleanup Race Condition
**File:** `frigate/storage.py:128-280`

`reduce_storage_consumption()` iterates recordings and deletes files, then bulk-deletes database rows. Another process (e.g., `RecordingCleanup`) could delete the same recordings between the file deletion and the DB cleanup. The `has_clip` update for events could mark events as "no clip" when clips still exist in the other process's view. The bulk delete query could target rows already removed, causing incorrect row counts.

**Impact:** Events incorrectly losing their clip association; potential UI confusion.

---

## Medium-Severity Issues

### 8. Hardcoded Paths Throughout
**Files:** Many (see below)

Key hardcoded paths:
| Path | File | Line |
|------|------|------|
| `/opt/frigate` | `frigate/const.py` | 4 |
| `/config` | `frigate/const.py` | 5 |
| `/media/frigate` | `frigate/const.py` | 8 |
| `/tmp/cache` | `frigate/const.py` | 16 |
| `/usr/local/lib/vec0` | `frigate/db/sqlitevecq.py` | 14 |
| `/tmp/cache/comms` | `frigate/comms/inter_process.py` | 16 |
| `/dev/shm` | `frigate/util/services.py` | 1169 |
| `/sys/devices/pci0000:00/...` | `frigate/util/services.py` | 514 |
| `/run/secrets` | `frigate/api/auth.py` | 316 |

**Impact:** Non-portable; hard to test outside Docker; breaks on systems with different filesystem layouts.

### 9. `UnboundLocalError` Risk in Proxy Parsing
**File:** `frigate/api/auth.py:256-260`

```python
try:
    network = ipaddress.ip_network(proxy)
except ValueError:
    logger.warning(f"Unable to parse trusted network: {proxy}")
trusted_proxies.append(network)  # network may be undefined!
```

The `except` block does not `continue`, so `network` is undefined when the loop continues to `append`. This raises `NameError` at runtime if any proxy address is malformed.

**Impact:** Auth endpoint crashes; all users locked out if a single bad proxy is configured.

### 10. Login Timing Side Channel
**File:** `frigate/api/auth.py:828-866`

The error message is uniform ("Login failed"), but `User.get_by_id` is called before `verify_password`. A non-existent user returns faster than a valid user with wrong password. This is a minor timing oracle for username enumeration.

**Impact:** Low — mitigated by `secrets.compare_digest` in `verify_password` for the password branch, but the username existence check is still observable.

### 11. Watchdog Detection Stuck — Restart Window Gap
**File:** `frigate/watchdog.py:121-128` and `frigate/object_detection/base.py:356`

The watchdog detects a stuck detector after 10 seconds of no `detection_start` update. However, `start_or_restart()` first calls `self.stop()` which has a **30-second** join timeout. During this 30-second window, the watchdog sees `detection_start == 0.0` (reset at line 356) and considers the detector healthy — but the old process is actually still being killed and the new process hasn't started yet.

**Impact:** Up to 30 seconds of missed detections during detector restart.

### 12. UntrackedSharedMemory Cleanup on Partial Init
**File:** `frigate/app.py:730-733`

```python
while len(self.detection_shms) > 0:
    shm = self.detection_shms.pop()
    shm.close()
    shm.unlink()
```

If `unlink()` fails (e.g., already unlinked by a subprocess), the exception propagates and prevents cleanup of remaining SHM segments. Missing `try/except` around `unlink()`.

**Impact:** Orphaned SHM segments in `/dev/shm` after crash restart.

### 13. ZMQ `send_data()` Returns Empty String on Error
**File:** `frigate/comms/inter_process.py:76-82`

```python
def send_data(self, topic: str, data: Any) -> Any:
    try:
        self.socket.send_json((topic, data))
        return self.socket.recv_json()
    except zmq.ZMQError:
        return ""
```

Returns `""` on failure, but callers (e.g., `track/object_processing.py`) may interpret an empty string differently than `None`. Some paths check `if not update` which treats `""` as falsy, but `""` could be a valid response in other contexts.

**Impact:** Subtle bugs where callers don't distinguish between "empty valid response" and "ZMQ error."

### 14. Misleading Peewee Model — `RecordingsToDelete`
**File:** `frigate/models.py:137-141`

```python
class RecordingsToDelete(Model):
    id = CharField(null=False, primary_key=False, max_length=30)
    class Meta:
        temporary = True
```

Has `primary_key=False` — Peewee requires a primary key for most ORM operations. Works only because raw SQL is used in `record/cleanup.py`, but the model definition is misleading.

**Impact:** If anyone tries to use ORM methods on this model, they'll get confusing errors.

---

## Low-Severity Issues

### 15. `_first_load_seen` Cache Grows Unbounded
**File:** `frigate/api/auth.py:298-304`

`_cleanup_first_load_seen()` is only called from `/profile` endpoint. If `/profile` is rarely accessed, the in-memory cache of anonymous access hashes grows without bound.

**Impact:** Slow memory leak (~72 bytes per unique client per week).

### 16. `set_file_limit()` Does Not Raise Hard Limit
**File:** `frigate/util/services.py:1141-1154`

Caps the soft limit at the current hard limit but never attempts to raise the hard limit. On constrained systems where the hard limit is also low (e.g., 4096), Frigate may still exhaust file descriptors under heavy camera load.

**Impact:** Potential file descriptor exhaustion on systems with very restrictive defaults.

### 17. Log Deduplication Mismatch on Long-Duration Repeats
**File:** `frigate/util/services.py:1073-1138`

`process_logs()` deduplicates by message content only, using `last_timestamp` in the dedup message. If messages repeat over many minutes/hours, the dedup message's timestamp represents only the first occurrence, which is misleading for troubleshooting.

**Impact:** Confusing log output for long-running repeating messages.

### 18. vec0 Extension Load Failure — No Clear Error
**File:** `frigate/db/sqlitevecq.py:17-25`

If `_load_vec_extension()` fails (missing library at `/usr/local/lib/vec0`), the connection is returned without the extension. Queries against vec0 virtual tables fail with opaque SQLite errors rather than a clear message.

**Impact:** Difficult debugging when semantic search silently doesn't work.

### 19. Module-Level `logging.warning` Instead of `logger.warning`
**File:** `frigate/comms/inter_process.py:49`

Uses `logging.warning(...)` (the module-level function) instead of `logger.warning(...)` (the module's logger instance defined at line 13). This bypasses the configured log format and handlers.

**Impact:** Inconsistent log formatting for ZMQ deserialization errors.

### 20. `restart_frigate()` Assumes Docker/S6 Layout
**File:** `frigate/util/services.py:34-41`

```python
proc = psutil.Process(1)
if proc.name() == "s6-svscan":
    proc.terminate()
else:
    os.kill(os.getpid(), signal.SIGINT)
```

Assumes PID 1 is `s6-svscan` (S6 overlay in Docker) or that SIGINT to self is sufficient. Breaks under `podman`, `systemd-nspawn`, raw `docker run --init`, or bare-metal deployments.

**Impact:** Frigate cannot cleanly restart on non-S6 environments.

---

## Summary Table

| # | Severity | File(s) | Issue |
|---|----------|---------|-------|
| 1 | Critical | `api/auth.py`, `api/fastapi_app.py` | JWT secret key derivation weakened for non-hex secrets |
| 2 | Critical | `api/auth.py` | JWT not invalidated until refresh after password change |
| 3 | Critical | `object_detection/base.py` | Stale SHM data race in remote detector |
| 4 | High | `comms/dispatcher.py` | 3× bare f-strings where `logger.error()` was intended |
| 5 | High | `__main__.py`, `app.py` | Pickle deserialization via SyncManager queues |
| 6 | High | `record/record.py`, `app.py` | Dual DB connection pools on same SQLite file |
| 7 | High | `storage.py` | Race between storage cleanup and recording cleanup |
| 8 | Medium | Multiple | Hardcoded filesystem paths |
| 9 | Medium | `api/auth.py` | `UnboundLocalError` risk in proxy IP parsing |
| 10 | Medium | `api/auth.py` | Username enumeration via login timing |
| 11 | Medium | `watchdog.py`, `base.py` | 30s detection gap during detector restart |
| 12 | Medium | `app.py` | Orphaned SHM segments if unlink fails |
| 13 | Medium | `comms/inter_process.py` | Ambiguous empty-string error return |
| 14 | Medium | `models.py` | Misleading Peewee model definition |
| 15 | Low | `api/auth.py` | Unbounded in-memory cache growth |
| 16 | Low | `util/services.py` | Soft file limit not raised to hard limit |
| 17 | Low | `util/services.py` | Log dedup timestamp misleading |
| 18 | Low | `db/sqlitevecq.py` | No clear error when vec0 extension missing |
| 19 | Low | `comms/inter_process.py` | Wrong logging function used |
| 20 | Low | `util/services.py` | Restart mechanism assumes Docker/S6 |
