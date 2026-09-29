# Vision Service

The off-Pi half of the robot's perception layer. Runs on the Mac mini; the
Raspberry Pi is the only client.

Why it exists, and why each endpoint is shaped the way it is, lives in
[`../docs/VISION-DESIGN.md`](../docs/VISION-DESIGN.md) Part 3. This file is the
contract: what to call, what comes back, and what to do when it doesn't.

---

## Quick start

```bash
cd vision_service
python3 -m venv .venv && source .venv/bin/activate

pip install -e '.[dev]'             # API + tests. No models, works anywhere.
pip install -e '.[models,dev]'      # + the real models. macOS arm64.

pytest                              # 36 tests, ~1s, no GPU needed
VS_BACKEND=fake ./run.sh            # serve stubs
./run.sh                            # serve for real (VS_BACKEND=real)
```

Interactive docs at `http://<mini>:8080/docs` once it's up — FastAPI generates
them from the same code that serves the requests, so they cannot drift.

---

## The two rules every endpoint follows

**1. Binary JPEG via `multipart/form-data`. Never base64.** Base64 inflates the
payload 33% and costs CPU on both ends to produce a bigger string. The Pi shares
its radio with a continuous Gemini Live audio uplink that has already died once
from contention; 33% is not available to give away.

**2. Every inference request carries `seq`, and every response echoes it.** The
Pi's `seq` is its monotonic capture counter, and it is the key to the frame stamp
holding that frame's capture time, gimbal angles and odometry pose. A response
whose `seq` the Pi doesn't recognise gets discarded — a result it cannot place in
time is worse than no result. Details in VISION-DESIGN.md Part 2.1.

---

## Endpoints

### `POST /v1/detect` — object detection

The hot path. Called continuously at `DETECT_HZ` (2 Hz).

**Request** — `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `image` | file | JPEG, ~640 px long edge (`DETECT_LONG_EDGE_PX`) |
| `seq` | int | Required. Echoed back |
| `min_confidence` | float | Default `0.4`, range 0–1 |

**Response** `200`

```json
{
  "seq": 10432,
  "detections": [
    { "class": "person", "confidence": 0.91, "bbox": [120.4, 88.1, 372.0, 359.2] },
    { "class": "potted_plant", "confidence": 0.77, "bbox": [401.0, 150.3, 502.7, 320.5] }
  ],
  "model_version": "yolov8n-mps",
  "image_size": [640, 360],
  "inference_ms": 18.4
}
```

`bbox` is `[x1, y1, x2, y2]` **in the pixel space of the image as sent**.
`image_size` echoes that size so the Pi never has to assume a scale — a bbox
silently in the wrong resolution looks exactly like bad camera calibration, and
you will spend an evening on it.

`model_version` changes when you swap weights. Log it on change (PLAN standing
rule #6). Note that branching on `"person"` couples you to the COCO label set;
keep that mapping in one place on the Pi.

```bash
curl -s -F image=@frame.jpg -F seq=1 -F min_confidence=0.4 \
     http://mini.local:8080/v1/detect | jq
```

---

### `POST /v1/faces/embed` — face embedding

Called **once per person track**, not per frame. The cascade that gates this on
"detect saw a person" is open continuously during person-following, which is the
exact case it was meant to protect against — so identity binds to the track
instead. VISION-DESIGN.md Part 6.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `image` | file | JPEG crop of the person/head region, from the `main` 720p stream |
| `seq` | int | Required |

**Response** `200`

```json
{
  "seq": 10433,
  "embedding": [0.0123, -0.045, "... 512 floats, L2-normalised ..."],
  "face_bbox": [12.0, 8.0, 148.0, 151.0],
  "quality": 0.86,
  "model_version": "insightface-buffalo_l",
  "inference_ms": 11.2
}
```

`quality` (0–1) combines face size, sharpness and detector confidence. It is not
calibrated — it only has to rank. **Discard below ~0.6 before spending a match
call.** `face_bbox` is relative to the crop you sent, not the full frame.

**`404`** means no face in the crop. That is a normal outcome for someone seen
from behind — not an error, and not worth retrying on the same frame.

---

### `POST /v1/faces/match` — identity lookup

No GPU, no model — a dot product against the gallery. Sub-millisecond.

**Request** — `application/json`

```json
{ "embedding": [0.0123, -0.045, "..."], "top_k": 1 }
```

**Response** `200`

```json
{
  "matches": [ { "label": "roshan", "similarity": 0.89 } ],
  "threshold": 0.75,
  "gallery_size": 12
}
```

Below `threshold`, treat as `unknown` — don't let the model force a guess. The
threshold is **returned rather than assumed** so you can tune it server-side
without a Pi deploy.

`matches` is best-score-per-person, not per-embedding: four photos of the same
face cannot crowd everyone else out of `top_k`. An empty gallery returns
`"matches": []`, not an error.

**Never act on a single frame's match.** Require two consecutive agreeing
matches before assigning identity to a track, then make it sticky for the life of
that track.

`422` on a malformed body, or when the embedding's dimension doesn't match the
gallery's — which means the face model changed and the gallery needs rebuilding.

---

### `POST /v1/faces/enroll` — add a known person

Admin/offline. Never on the hot path.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `label` | string | 1–64 chars |
| `images` | file[] | Repeat the field per image. Max 16 |

**Response** `200`

```json
{ "label": "roshan", "embeddings_added": 4, "images_rejected": 1, "gallery_size": 12 }
```

Images with no detectable face are counted in `images_rejected` rather than
failing the call — one bad photo in a batch of ten shouldn't lose the other nine.
`422` only when *every* image was rejected.

```bash
curl -s -F label=roshan \
     -F images=@a.jpg -F images=@b.jpg -F images=@c.jpg \
     http://mini.local:8080/v1/faces/enroll | jq
```

Aim for 4–8 images per person across varied lighting and angle. More than that
adds little; fewer makes the threshold delicate.

---

### `GET /v1/faces` · `DELETE /v1/faces/{label}` — gallery admin

```json
{ "labels": ["roshan", "sam"], "embeddings": 12, "people": 2 }
```

`DELETE` forgets a person entirely and returns `embeddings_removed`; `404` if
they weren't enrolled. Labels only — embeddings are never browsable.

---

### `POST /v1/depth` — monocular depth ⚠️ *fallback only*

**Prefer the lidar.** The camera gives an accurate *bearing*; `/scan` gives an
accurate *metric range* at that bearing, locally, at 6 Hz, for free. Call this
only when the scan sees past the object because it sits above or below the
lidar's horizontal plane. For a person this basically never happens — the lidar
sees legs, which is the range you want.

**Request** — `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `image` | file | JPEG |
| `seq` | int | Required |
| `points` | string | JSON array of `[cx, cy]` pixel points. Max 64 |

`points` are bbox centres, so the response is sparse rather than a full depth
map — much smaller payload, and you only ever want a handful of values.

**Response** `200`

```json
{
  "seq": 7,
  "depths_m": [1.8, 3.2],
  "confidence": [0.7, 0.5],
  "metric": false,
  "model_version": "Depth-Anything-V2-Small-hf",
  "inference_ms": 142.0
}
```

> **Check `metric`, not the field name.** Depth Anything V2 returns *relative
> inverse depth*: the values order correctly but **are not metres**, despite
> `depths_m`. Using them as distances will place objects wrongly — at 3 m, a 20%
> error is ±0.6 m against a 1.5 m standoff radius, and the approach looks broken
> for a reason that has nothing to do with the planner. Swap in a metric model
> (Metric3D, Depth Pro, UniDepth) and set `metric = True` on the backend if you
> need real distances here.

---

### `GET /healthz` · `GET /v1/models` — ops

```json
{
  "status": "ok",
  "backend": "real",
  "versions": { "detect": "yolov8n-mps", "face": "insightface-buffalo_l", "depth": "Depth-Anything-V2-Small-hf" },
  "gallery": { "embeddings": 12, "people": 2 },
  "queue_depth": 0,
  "uptime_s": 5821.3
}
```

Two fields worth alerting on. **`backend: "fake"` in production means the robot
is seeing invented furniture** — the service logs a loud warning at startup, but
check it here too. **`queue_depth` climbing** means requests are arriving faster
than the GPU retires them.

---

## Errors

Every failure returns the same shape, so the Pi parses one thing:

```json
{ "error": "image_too_large", "detail": "..." }
```

| Status | `error` | What the Pi should do |
|---|---|---|
| `400` | `image_undecodable` | Drop the frame. Corrupt or truncated upload |
| `404` | — | No face in crop. Normal; don't retry this frame |
| `413` | `image_too_large` | Bug on the Pi side — check `DETECT_LONG_EDGE_PX` |
| `422` | — | Malformed request. A bug, not a transient |
| `503` | `busy` | **Back off.** Honours `Retry-After`. Drop the frame, don't queue |
| `504` | `inference_timeout` | Model stuck or machine throttled. Drop the frame |

`503` is deliberate and it is the server half of PLAN standing rule #8: there is
one GPU, so inference is serialized through a single worker thread, and requests
arriving while `VS_MAX_QUEUE_DEPTH` are already waiting are **refused rather than
queued**. A refusal the Pi can react to immediately beats an answer that arrives
after the Pi has stopped caring. The Pi's own policy — one request in flight,
drop don't queue, discard stale results — is in VISION-DESIGN.md Part 4.3, and
the two halves are designed to agree.

---

## Configuration

All environment variables. Defaults match VISION-DESIGN.md's appendix.

| Variable | Default | Notes |
|---|---|---|
| `VS_HOST` / `VS_PORT` | `0.0.0.0` / `8080` | |
| `VS_BACKEND` | `auto` | `real` \| `fake` \| `auto`. **Set `real` in production** |
| `VS_DEVICE` | `mps` | Falls back to `cpu` with a warning if unavailable |
| `VS_DETECT_MODEL` | `yolov8n.pt` | Downloaded on first run |
| `VS_DETECT_IMGSZ` | `640` | Match what the Pi sends |
| `VS_FACE_MODEL` | `buffalo_l` | InsightFace pack |
| `VS_DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | |
| `VS_GALLERY_PATH` | `~/.robot-vision/gallery.npz` | Personal data. Not in the repo |
| `VS_MATCH_THRESHOLD` | `0.75` | Returned in every match response |
| `VS_MAX_UPLOAD_BYTES` | `4194304` | Sanity bound; real frames are ~30 KB |
| `VS_MAX_ENROLL_IMAGES` | `16` | Per enroll call |
| `VS_MAX_QUEUE_DEPTH` | `4` | Above this, refuse with `503` |
| `VS_INFERENCE_TIMEOUT_S` | `10.0` | Server-side; the Pi's own is 0.6 s |

`VS_BACKEND=auto` prefers real models and degrades to fake with a warning, so a
fresh clone runs. Production sets `real` so a missing dependency is a startup
failure rather than a robot that confidently sees furniture that isn't there.

---

## Running it permanently

`deploy/com.roshan.visionservice.plist` is a launchd agent with `KeepAlive`, a
restart throttle, and `caffeinate` in `run.sh`.

```bash
cp deploy/com.roshan.visionservice.plist ~/Library/LaunchAgents/
# edit the two absolute paths inside first
launchctl load -w ~/Library/LaunchAgents/com.roshan.visionservice.plist

launchctl list | grep visionservice
tail -f /tmp/vision-service.log
launchctl kickstart -k gui/$(id -u)/com.roshan.visionservice     # restart
```

**`caffeinate -i` is not decoration.** A sleeping mini looks exactly like bad
wifi from the Pi's side, and that is a miserable thing to debug. A LaunchAgent
(not a LaunchDaemon) is correct here: it runs in your GUI session, which is what
lets it reach the GPU.

---

## Architecture

```
app.py          routes, validation, error shapes — the only file that knows HTTP
runner.py       one worker thread + bounded waiting. Serializes GPU access
gallery.py      labelled embeddings on disk, cosine match in numpy
images.py       bytes -> RGB ndarray, with the size limit enforced early
schemas.py      response models. This IS the contract
backends/
  base.py       the Protocol every backend satisfies
  real.py       Ultralytics YOLO, InsightFace, Depth Anything V2
  fake.py       deterministic stubs. No models, no GPU, no network
```

**The backend seam is the reason the test suite runs anywhere.** `FakeBackend`
satisfies the same protocol as the real one, so every route, validation path and
error branch is covered on a laptop in under a second — and you can develop the
Pi client on a train. Fake outputs are derived from image content, so they're
stable across runs but differ between images, which is what a test needs.

Two deliberate choices worth knowing:

**Inference is serialized through one worker thread**, not a lock. There is one
GPU; two inferences at once are slower and less predictable, not faster. Using a
single-worker executor rather than a lock also avoids the nasty failure where a
timeout releases a lock while its thread still owns the GPU.

**The gallery is not a vector database.** A household gallery is a few people
with a few embeddings each; a brute-force dot product over an `(N, 512)` array is
microseconds. An index would add a dependency and a failure mode to save nothing
measurable. Writes are atomic via a temp file, and a corrupt gallery logs an
error and starts empty rather than stopping the service — degraded recognition,
not a dead robot.

---

## Licensing

Worth knowing before this becomes anything but a robot in your house:

- **Ultralytics** (YOLOv8) is **AGPL-3.0**. Fine for personal use; it obliges
  source disclosure for network-served derivative works.
- **InsightFace** pretrained packs are **research / non-commercial**.

Both sit behind the `Detector` / `FaceEncoder` protocols in `backends/base.py`,
so swapping them is a contained change — one file, no route touched. If you ever
need permissive: export YOLO to ONNX and run it through `onnxruntime` with the
CoreML execution provider.

---

## Performance notes

Expect on an M-series mini, `VS_DEVICE=mps`, steady state:

| Endpoint | Typical |
|---|---|
| `/v1/detect` @ 640 px | 15–25 ms |
| `/v1/faces/embed` | 10–20 ms |
| `/v1/faces/match` | < 1 ms |
| `/v1/depth` | 100–200 ms |

The **first** inference after startup is much slower — MPS compiles kernels on
first use — which is why `YoloDetector.__init__` runs a warm-up pass. Without it
the first real frame blows the Pi's 0.6 s timeout.

`inference_ms` is in every response. Log it on the Pi alongside the wall-clock
round trip: the difference is your network, and knowing which half is slow is
worth more than either number alone. Phase V1 in VISION-DESIGN.md exists
specifically to measure this before anything is built on top of it.
