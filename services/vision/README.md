# soccer-vision-service

A standalone microservice that ingests **video/frames** of a soccer match, runs
a configurable pipeline of **CV kernels** (segmentation, homography, density
estimation, object detection, tracking), and emits canonical events conforming
to the [`soccer_model.MatchEventStream`](../../soccer_model/schema.py) schema
from the main project.

## Why a separate service?

The core `soccer-world-model` library is intentionally small (numpy + scipy +
pydantic). Computer vision drags in heavy native deps (OpenCV, ffmpeg, CUDA,
PyTorch). Running CV in a dedicated container lets you:

- scale GPU-bound vision pods independently from the analytics layer
- update model checkpoints without redeploying the core library
- swap detector implementations behind a stable HTTP / WebSocket contract

The contract between services is the existing `MatchEventStream` JSON schema —
so anything the vision service emits flows transparently into the rest of the
analytics stack.

---

## Layout

```
services/vision/
├── pyproject.toml               # soccer-vision-service package
├── Dockerfile
├── docker-compose.yml
├── soccer_vision/
│   ├── api/                     # FastAPI app
│   │   ├── server.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── kernels/                 # pure-numpy image / spatial kernels
│   │   ├── base.py              # Kernel ABC
│   │   ├── gaussian.py          # 1D / 2D Gaussian + separable conv
│   │   ├── kde.py               # 2D Gaussian KDE / density / heat maps
│   │   ├── pitch_mask.py        # HSV-based green-pitch segmentation
│   │   └── homography.py        # 8-DoF projective transform (DLT)
│   ├── detection/               # ML-backed detection (lazy imports)
│   │   ├── base.py              # Detector ABC + Detection model
│   │   └── opencv_stub.py       # color-based ball stub (no ML)
│   ├── tracking/
│   │   └── tracker.py           # SORT-like multi-object tracker stub
│   ├── streaming/
│   │   └── source.py            # FrameSource ABC (file, rtsp, websocket, …)
│   ├── events/
│   │   └── emitter.py           # detections → MatchEventStream events
│   ├── pipeline.py              # orchestrates source → detect → track → event
│   └── config.py                # pydantic-settings runtime config
└── tests/
    ├── test_kernels.py
    ├── test_homography.py
    ├── test_kde.py
    ├── test_emitter.py
    └── test_api.py
```

---

## Install

```bash
# from the repo root
pip install -e ./services/vision[dev]

# add the ML extras when you want real object detection:
pip install -e ./services/vision[dev,ml,streaming]
```

The base install (no `ml` extra) gives you the API surface and all pure-numpy
kernels (KDE, homography, Gaussian filtering). The `ml` extra adds OpenCV,
PyTorch, and YOLO for real player/ball detection.

## Run

```bash
soccer-vision                                  # uvicorn on :8088
# or
uvicorn soccer_vision.api.server:app --reload
```

## Docker

```bash
docker compose -f services/vision/docker-compose.yml up --build
```

## Contract

### REST

| Method | Path                       | Purpose                                          |
|--------|----------------------------|--------------------------------------------------|
| GET    | `/v1/healthz`              | liveness probe                                   |
| GET    | `/v1/readyz`               | readiness (checks ML model load if `ml` extra)   |
| POST   | `/v1/sessions`             | create analysis session (homography, lineup)     |
| GET    | `/v1/sessions/{sid}`       | inspect session                                  |
| POST   | `/v1/frames/analyze`       | analyse a single frame, return events            |
| WS     | `/v1/stream/{session_id}`  | bidirectional: frames in, events out             |

### Events emitted

Every event posted out of the vision service is a valid
`soccer_model.schema.*Event` (i.e. it round-trips through
`soccer_model.validate_event_stream`).

Pitch coordinates emitted by the vision service are always in the FIFA
reference frame (`x ∈ [0,105]`, `y ∈ [0,68]`) — the camera-to-pitch
homography handles the transform from pixel space.

---

## Kernels

The `kernels/` sub-package is the **pure-numpy** layer. It has no ML or
OpenCV dependency and is fully unit-tested.

- `gaussian.GaussianKernel1D / 2D` — separable 2-D Gaussian for blurring,
  smoothing tracking residuals, etc.
- `kde.GaussianKDE2D` — 2-D kernel density estimate over the pitch with
  optional masked region (e.g. only count touches in the final third).
  Used for heat maps, pressing density, defensive shape.
- `pitch_mask.HSVPitchMask` — green-channel segmentation for isolating
  the pitch from stands/people in a frame. Cheap, deterministic.
- `homography.PlanarHomography` — solves the 8-DoF projective transform
  given ≥ 4 pixel↔pitch correspondences (DLT). Lets the rest of the
  pipeline work in pitch metres regardless of camera angle.

These are deliberately useful **without** any ML stack — you can compute
heat maps and pitch transforms today, and slot detectors in later.

---

## Detection / tracking (scaffold)

The `detection/` and `tracking/` modules currently expose ABCs and one
trivial reference implementation (`OpenCVBallStub`). Plug in YOLO or a
custom detector by implementing `Detector.detect(frame) -> list[Detection]`.

This boundary is intentional: the ML choice (YOLOv8, RT-DETR, custom) is a
deployment-time decision, not a contract-time one.

---

## See also

- [`../../docs/EVALUATION.md`](../../docs/EVALUATION.md) — full codebase
  evaluation that motivated this service.
- [`../../README.md`](../../README.md) — core library docs.
