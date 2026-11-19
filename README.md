# Vision Service

The off-Pi half of the robot's perception layer. Runs on the Mac mini; the
Raspberry Pi is the only client.

Why it exists, and why each endpoint is shaped the way it is, lives in
`docs/VISION-DESIGN.md` Part 3 in the Robot-Computer-Vision repo (as do the
`PLAN.md` rules referenced below). This file is the contract: what to call,
what comes back, and what to do when it doesn't, plus how to deploy it.

---

## Quick start

On the Mac mini, from the repo root (needs Python 3.10+, e.g. `brew install python`):

```bash
make deploy        # create .venv, install everything, run the tests, start the service
make status        # is it running and healthy?
```

That's the whole setup. After that, **every time you change the code:**

```bash
make deploy
```

Details, and what each step does, in [Deploying](#deploying).

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

The accepted photos are also saved to `faces/<label>/photo_{i}.png` with a
`bbox_{i}.txt` (`x1 y1 x2 y2` of the face), by a background task after the
response. `label` becomes a folder name, so it must match `[A-Za-z0-9_-]{1,64}`
(`422` otherwise).

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

### Explore mode: plants, `/v1/ask`, SLAM

Used by the robot's explore mode (`Robot-RPi-Core/docs/EXPLORE-DESIGN.md`).
Full request/response shapes: `http://<mac>:8080/docs`.

| Endpoint | What it does |
|---|---|
| `POST /v1/plants/embed` | Plant crop → DINOv2 embedding |
| `POST /v1/plants/match` | Embedding (+ map position `[x, y]` or `[x, y, height]`) → best plants, each with similarity, its stored position (with height, when measured) **and** floor distance from where it was last seen |
| `POST /v1/plants/enroll` | Photos + meta → a new `plant_NNN`, or a new visit when `label` is an existing plant. Embeddings are stored before it answers; the photos are written by a background task to `plants/map_{index}/plant_{NNN}/photo_{i}.png` + `bbox_{i}.txt` (`x1 y1 x2 y2` of the potted plant, empty if none found). `map_{index}` is bumped by each new `POST /v1/slam/reset` |
| `GET /v1/plants` · `GET /v1/plants/{label}` · `GET /v1/plants/{label}/{file}` · `DELETE /v1/plants/{label}` | Register admin; delete moves photos to `_deleted/` |
| `POST /v1/ask` | Images + question → answer from Gemini, or a local VLM via Ollama. `json=true` also parses the answer |
| `POST /v1/slam/keyframe` | Odometry pose + rays (+ from the 8x8 ToF: `clear` free-space rays, `low` obstacles, `drops`, `cloud` 3-D points) → corrected pose, `map_to_odom`, `loop` (see below), next unexplored goal and path, with `goal.clear_m`: how close each leg passes to anything mapped (the robot drives roomy legs whole). Everything is saved in `maps/<name>.npz` |
| `POST /v1/slam/plan` · `POST /v1/slam/viewpoints` | Path between two points (+ `clear_m` per leg); reachable poses around a plant. Paths keep 0.15 m more room from walls where the map allows (`pref_clearance`), else take the tightest way |
| `GET /v1/slam/walls` | Straight walls fitted to the map, each with its length, and `extent`: the explored area's length and width along its main directions (several rooms explored together make one extent) |
| `POST /v1/slam/reset` · `POST /v1/slam/save` · `GET /v1/slam/state` | Runs and saved maps (`reset` with `load: true` continues a saved map from a given start pose) |
| `GET /v1/slam/map.png` | The map: free white, walls black, low obstacles orange, drops (stairs down) magenta, robot trail blue, plants red with their height (plants on one stand listed together) |

SLAM and `/v1/ask` run in worker threads, never through the GPU runner, so a
slow map update or a 10 s VLM answer never delays `/v1/detect`.

**Loop closure.** When a keyframe is back within 1.5 m of one taken at least 5
keyframes and 3 m of driving ago, its scan is matched against the old keyframes
alone. If they agree (fit at least 0.5) and the robot's chain of poses has drifted,
the correction is spread along the keyframes in between, and the map is rebuilt
from their stored readings. The keyframe answer's `loop` says what happened:
`{"closed": true, "with_kf", "score", "correction_m", "correction_deg"}` or
`{"closed": false}`. It works within one run; a run continuing a saved map
rebuilds on top of the loaded map.

**Plant positions** are the median of every visit's measured position (one bad
ranging cannot drag it), with `position_spread_m` saying how far the visits
disagree.

### Follow mode: `POST /v1/follow/step` measurements

Each `tracking` answer carries what the robot needs to decide its own move:
- `bearing_deg`: in the robot's frame, with the head's pan added; + is left.
- `distance_m` with `distance_from`, and `distance_sigma_m`, how far off it may be:
  - ToF zones: about 5 cm
  - feet on the floor: d²/h per degree of tilt error
  - face size: 15 %
  - box width: 30 %
- `ground_xy`: where they stand, metres ahead and left of the robot when the frame was taken.
- `confidence`: 1.0 for a face match, else how well their clothes match.

`turn_deg` and `drive_m` are still sent. The robot uses them only while
searching; while tracking, its own person tracker decides the turn and the leg.

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

The field worth alerting on is **`queue_depth` climbing**: requests are arriving
faster than the GPU retires them. `backend` is always `"real"` in the running
service.

### Debug uploads (`debug_api.py`)

Not inference: the robot's `debug_mode` (robot.yaml `/voice debug_mode: true`) sends the camera frames and ToF grids its tasks collect, at most 2 a second, and they are stored exactly as sent for Monitoring-Service (`tof_camera_overlaid.py`) to show.

| Endpoint | Body | Stores / returns |
|---|---|---|
| `POST /v1/debug/frame` | form field `meta` (JSON: `session`, `n`, `ts`, `frame`, `tof.rows` in metres, `geometry`, `extras`), optional `image` (JPEG) | `debug/<session>/<n>.jpg` and `.json` (plus `received`, this Mac's clock); `debug/latest.json` points at the newest |
| `GET /v1/debug/latest` | | the newest upload's meta |

Folder: `VS_DEBUG_DIR` (default `debug/` next to the code, gitignored). Only the newest `VS_DEBUG_KEEP` sessions (default 20) are kept; starting a new one deletes the oldest beyond that.

---

## Global navigation (`nav_api.py`)

The robot drives itself locally (DWA on a rolling costmap, `robot_core/nav` on the Pi); this
service is its long-horizon layer.

| Endpoint | What |
|---|---|
| `POST /v1/nav/frame` | the camera stream (~10 fps): `image`, `t`, `pose` (JSON x, y, heading), `pan_deg`, `tilt_deg` (real) -> visual SLAM |
| `POST /v1/nav/plan` | `{pose, goal, radius_m, local}` -> `{path}`: Theta* waypoints in the Pi's odometry frame; `local` (the Pi's costmap) is folded into the map first |
| `GET /v1/nav/state` | map size, plans, SLAM backend/tracking/scale, `odom_to_map` |
| `GET /v1/nav/map.png` | the map: occupied black, free white, unknown grey, camera-found furniture orange, last path blue, robot red |
| `POST /v1/nav/reset` | a new map |

Visual SLAM is ORB-SLAM3 (monocular). Its metric scale comes from the ToF: the Pi sends the ToF grid
nearest each frame, and the depths of the points ORB-SLAM3 tracks are compared with the ranges of the
zones that see a flat surface there (corrected for each zone reporting the nearest point in its cone).
Odometry only levels and places the map, and gives the scale until the ToF has 8 agreeing views. It is optional: install it
with `bash deploy/install_orbslam3_macos.sh` (Homebrew deps, Pangolin, ORB-SLAM3, vocabulary,
Python bindings, then a check). Until then `/v1/nav/state` shows `"backend": "none"` and the map
runs on the Pi's odometry. Settings: `VS_ORB_VOCAB`, `VS_VSLAM=0` to turn it off,
`VS_CAMERA_HFOV_DEG`, `VS_HEAD_X_M` / `VS_HEAD_Y_M` / `VS_HEAD_HEIGHT_M` (match robot.yaml).
Calibrate the camera for best results: the intrinsics written to `~/.robot-vision/vslam/camera.yaml`
are estimated from the field of view. ORB-SLAM3 is GPLv3.

While a robot debug session is receiving uploads, everything this layer computes is saved beside
them in `debug/<session>/nav/` (`nav_record.py`): `nav.jsonl` (SLAM status, every tracked frame,
every SLAM pose and the base pose derived from it, each `odom_to_map` correction, the odometry fit
with its point pairs, new-map events, every plan with its path, margin and timing, and camera-placed
objects) and `map-NNNNNN.npz` snapshots of the global map (at most one per 2 s). Monitoring-Service's
`tof_camera_overlaid.py --nav --session S` replays them next to the robot's own uploads.

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

All environment variables. Defaults match VISION-DESIGN.md's appendix. To change
any of them, put `KEY=value` lines in a `.env` file at the repo root (start from
`.env.example`; plain values, no quotes), then `make restart`. `.env` is
gitignored, and both `run.sh` and the Makefile read it.

| Variable | Default | Notes |
|---|---|---|
| `VS_HOST` / `VS_PORT` | `0.0.0.0` / `8080` | |
| `VS_DEVICE` | `mps` | `mps` \| `cuda` \| `cpu`. Falls back to `cpu` with a warning if unavailable |
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
| `VS_PLANT_MODEL` | `facebook/dinov2-small` | Plant identity embedding (downloaded on first start) |
| `VS_PLANT_DIR` | `./plants` (next to the code) | Plant register: embeddings + `map_{index}/plant_{NNN}/` photo folders |
| `VS_FACE_PHOTO_DIR` | `./faces` (next to the code) | Enrolled face photos: `<name>/photo_{i}.png` + `bbox_{i}.txt` |
| `VS_PLANT_MATCH_THRESHOLD` | `0.62` | Tune after a few real plants |
| `VS_VLM_PROVIDERS` | `gemini,ollama` | Tried in order by `/v1/ask` |
| `GEMINI_API_KEY` | — | Enables the `gemini` provider |
| `VS_GEMINI_MODEL` | `gemini-3.6-flash` | |
| `VS_OLLAMA_URL` / `VS_OLLAMA_MODEL` | `http://127.0.0.1:11434` / `qwen3-vl:8b` | Local, 4-bit: `ollama pull qwen3-vl:8b` |
| `VS_VLM_TIMEOUT_S` | `45` | |
| `VS_SLAM_DIR` | `./maps` (next to the code) | Saved maps: `<name>.npz` (reloadable) + `<name>.png`, rewritten after every keyframe |
| `VS_SLAM_SIZE_M` / `VS_SLAM_RES_M` | `24` / `0.05` | Map extent and cell size |
| `VS_ROBOT_RADIUS_M` | `0.18` | Paths keep this far (+5 cm) from walls |

The service always loads the real models. A missing dependency or a model that
won't load is a startup failure (visible in `make logs`), never a silent
degradation.

---

## Deploying

The service runs in the background as a **launchd agent**, macOS's built-in
service manager (think of it as `docker run --restart unless-stopped` without the
container). There is no build step: the code is installed into `.venv` in
editable mode, so the running service uses the files in this folder directly.
"Deploying" a change just means reinstalling dependencies if they changed,
running the tests, and restarting.

### After every change

```bash
make deploy
```

That runs, in order, stopping at the first failure:

1. **`make install`**: creates `.venv` and installs `.[models,dev]`. Only does
   work when `pyproject.toml` has changed since the last install, so it's
   instant most of the time.
2. **`make test`**: the test suite. A failing test stops the deploy, and the
   old version keeps running.
3. **`make start`**: stops the running service, starts the new code, then
   waits for `/healthz` to answer (up to 180 s, since the first start
   downloads model weights) and prints it. If it never comes up, it prints
   the end of the log instead.

### Everyday commands

| Command | What it does |
|---|---|
| `make deploy` | Install if needed → test → restart → wait until healthy |
| `make start` / `make restart` | Restart without reinstalling or testing |
| `make stop` | Stop it. Stays stopped until `make start` or your next login |
| `make status` | launchd state (running? pid? crash count?) + `/healthz` |
| `make logs` | Follow the log. `Ctrl-C` to stop following |
| `make run` | Run in this terminal instead (for debugging; `make stop` first) |
| `make test` | Just the tests |
| `make reinstall` | Delete `.venv` and install from scratch |
| `make uninstall` | Stop it and remove it from launchd entirely |
| `make help` | List all of the above |

### What's where

| Thing | Location |
|---|---|
| Python environment | `.venv/` in this folder |
| Your settings | `.env` in this folder (optional, see [Configuration](#configuration)) |
| launchd job | `~/Library/LaunchAgents/com.roshan.visionservice.plist`, generated from `deploy/com.roshan.visionservice.plist.in` by `make start`, so its paths always point at this checkout |
| Log | `~/Library/Logs/Vision-and-Planning/vision-and-planning.log`. The previous run's log is kept as `.log.1`. One timestamped line per request: client, endpoint, status, time taken, the inputs (uploaded images as name, bytes and pixel size; form fields) and the answer (embeddings shortened to `<512 numbers>`); see `vision_service/reqlog.py` |
| Face gallery | `~/.robot-vision/gallery.npz` (`VS_GALLERY_PATH`) |
| Model weights | Downloaded on first start: YOLO into this folder (gitignored), InsightFace into `~/.insightface`, Depth Anything into `~/.cache/huggingface` |

### How it stays up

- **Starts at login** and **restarts on crash** (launchd `KeepAlive`, at most
  once every 10 s so a crash loop can't spin the CPU). `make status` shows
  `runs` and `last exit code` if it has been restarting.
- **Keeps the Mac awake** while it runs: `run.sh` starts `caffeinate` tied to
  the server's process, so it ends when the server does. A sleeping mini looks
  exactly like bad wifi from the Pi's side, and that is a miserable thing to
  debug. Set the mini to log in automatically after a power cut, or the service
  won't start until someone logs in.
- A LaunchAgent (not a LaunchDaemon) is deliberate: it runs as you, in your
  login session, which is what lets it reach the GPU.

### When something's wrong

| Symptom | Look at |
|---|---|
| `make deploy` stops at tests | The failing test's output. Nothing was restarted |
| "not healthy after 180s" | The log lines it printed. Usually a model failing to load or a missing package; `make reinstall` fixes the latter |
| "Port 8080 is already in use by: ..." | Something else is listening on 8080 — `make start` refuses to start and prints what it is. Usually a `./run.sh`/`make run` left open in another terminal, or an old copy of the service. `kill <PID>` it, or set `VS_PORT` in `.env` (and on the Pi) |
| `make status` shows `runs` climbing | It's crash-looping. `make logs` |
| Pi gets timeouts, service looks fine | `make health`: check `queue_depth` and `versions.detect` ends in `-mps`, not `-cpu` |

---

## Architecture

```
Makefile            install / test / deploy / start / stop / logs. `make help`
run.sh              what launchd runs: loads .env, keeps the Mac awake, starts uvicorn
deploy/             launchd job template (rendered by `make start`)
vision_service/
  app.py            routes, validation, error shapes — the only file that knows HTTP
  runner.py         one worker thread + bounded waiting. Serializes GPU access
  gallery.py        labelled embeddings on disk, cosine match in numpy
  images.py         bytes -> RGB ndarray, with the size limit enforced early
  schemas.py        response models. This IS the contract
  settings.py       every VS_* environment variable and its default
  backends/
    base.py         the Protocol every backend satisfies
    real.py         Ultralytics YOLO, InsightFace, Depth Anything V2
tests/
  stub_backend.py   test-only stand-in for real.py. No models, no GPU
```

**The backend seam is the reason the test suite runs anywhere.** The tests swap
in `tests/stub_backend.py`, which satisfies the same protocols as `real.py`, so
every route, validation path and error branch is covered in about a second with
no models installed. That stub exists only inside the tests; the service itself
has no way to select it.

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
