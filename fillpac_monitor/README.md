# FillPac Operator Working-Zone Monitor

Production-ready CPU-based monitor that tracks whether an operator is
present at a station and whether their hands are actively working inside a
defined zone, using YOLO pose (person tracking + skeleton) and MediaPipe
HandLandmarker (finger-level detail). Status changes are logged to SQL
Server (manageable via SSMS) plus rotating file logs.

Status values: `NO_OPERATOR`, `OPERATOR_PRESENT`, `NOT_WORKING`, `WORKING`.

## Project layout

```
fillpac_monitor/
├── main.py                     # entry point
├── config/
│   └── config.yaml             # all non-secret settings
├── .env.example                # copy to .env, fill in secrets
├── requirements.txt
├── src/
│   ├── app.py                  # orchestrates everything
│   ├── config_loader.py        # YAML + .env merge, ${VAR} expansion
│   ├── logging_setup.py        # rotating file logs + optional DB sink
│   ├── camera/capture_thread.py# RTSP/USB open, warmup, threaded reads, reconnect
│   ├── detection/
│   │   ├── pose_model.py       # YOLO pose wrapper
│   │   └── hand_model.py       # MediaPipe HandLandmarker wrapper
│   ├── zones/zone_manager.py   # interactive zone drawing + JSON persistence
│   ├── monitor/
│   │   ├── operator_monitor.py # per-frame processing + overlay drawing
│   │   └── state_machine.py    # debounced status confirmation + DB events
│   └── db/
│       ├── db_manager.py       # async batched SQL Server writer
│       └── schema.sql          # run this in SSMS first
├── scripts/
│   └── check_db_connection.py  # verify DB connectivity/schema before running
├── zones_config/                # saved zone polygons per station (auto-created)
├── logs/                        # rotating log files (auto-created)
└── models/                      # put yolo26n-pose.pt here; hand_landmarker.task auto-downloads
```

## Setup

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`pyodbc` requires the Microsoft ODBC Driver for SQL Server installed on the
machine. On Windows this is usually already present if SSMS is installed;
otherwise install "ODBC Driver 18 for SQL Server" from Microsoft.

### 2. Model weights

Place your YOLO pose weights at `models/yolo26n-pose.pt` (path is
configurable via `models.pose_model_path` in `config.yaml`). The MediaPipe
hand model downloads automatically on first run if not present.

### 3. Database (SSMS)

1. Open SSMS, connect to your SQL Server instance.
2. Open and execute `src/db/schema.sql` — it creates the `FillPacMonitor`
   database, all tables, indexes, and two reporting views:
   - `dbo.vw_CurrentStatus` — latest status per station.
   - `dbo.vw_DailyStatusSummary` — total WORKING / NOT_WORKING / NO_OPERATOR
     seconds per station per day.
3. Create a SQL login for the service account (or use Windows auth) and
   grant it `db_datareader` + `db_datawriter` on `FillPacMonitor`.

### 4. Configure secrets

```bash
cp .env.example .env
```

Fill in `CAMERA_USER`, `CAMERA_PASS`, `DB_SERVER`, `DB_NAME`, `DB_USER`,
`DB_PASSWORD` (or set `DB_TRUSTED_CONNECTION=true` for Windows auth).
`.env` is never read by config.yaml directly — it's loaded into the process
environment and `${VAR}` placeholders in `config.yaml` are substituted.

### 5. Verify DB connectivity

```bash
python scripts/check_db_connection.py
```

### 6. Run

```bash
python main.py
```

On first run per station, two interactive windows open for you to click out
the **operator zone** and the **hand work zone** polygons (Enter/Space to
finish, R to reset, Esc to cancel). The zones are then saved to
`zones_config/<station_id>.json` and reused automatically on subsequent
runs — you won't be prompted again unless you press `R` during the run or
delete the JSON file. Press `Q` to quit.

For headless/server deployment, set `display.show_window: false` in
`config.yaml` — in that case zones must already exist in `zones_config/`
(draw them once locally, or set `zones.interactive_draw_on_first_run: false`
to fall back to the fractional defaults).

## How status is determined

- **NO_OPERATOR**: no person detected inside the operator zone (after a
  short grace period defined by `monitor.operator_absence_timeout_sec`).
- **OPERATOR_PRESENT**: a person is in the operator zone but no hand is
  actively working in the work zone yet (e.g. just walked in).
- **NOT_WORKING**: operator present, hands not moving meaningfully inside
  the work zone.
- **WORKING**: at least one hand has any of its 21 landmarks inside the
  work zone **and** the average landmark displacement across the recent
  frame window exceeds `monitor.hand_movement_threshold_px` — this counts
  finger/joint motion even when the wrist itself is nearly stationary.

## Debounced logging (why the DB doesn't flood)

Raw per-frame status is noisy (a single missed detection can flicker
WORKING → NOT_WORKING → WORKING). `StatusStateMachine` only writes a
`CHANGE` row to `dbo.StatusEvents` once a new status has been observed
continuously for `monitor.status_debounce_sec` (default 1.5s). While a
state persists, a `HEARTBEAT` row is written every
`monitor.heartbeat_interval_sec` (default 60s) so long stable periods are
still visible without waiting for the next transition.

All DB writes go through a background thread with a bounded queue and
retry/backoff — if SQL Server is briefly unreachable, the vision loop is
never blocked and events queue up until the connection recovers.

## Logs

- `logs/fillpac_monitor.log` — all INFO+ messages, rotates at 10MB × 10 files.
- `logs/fillpac_monitor.errors.log` — WARNING+ only.
- `dbo.SystemLogs` — WARNING+ messages mirrored to the database (set
  `logging.also_log_to_db: false` to disable).

## Useful queries (SSMS)

```sql
-- Current status of every station
SELECT * FROM dbo.vw_CurrentStatus;

-- Today's working time vs idle time per station
SELECT * FROM dbo.vw_DailyStatusSummary
WHERE EventDate = CAST(SYSUTCDATETIME() AS DATE);

-- Raw event history for one station, most recent first
SELECT TOP 200 * FROM dbo.StatusEvents
WHERE StationID = 'FILLPAC-01'
ORDER BY EventTimeUtc DESC;
```

## Running as a service

- **Windows**: wrap `python main.py` with NSSM or Task Scheduler
  (run at startup, restart on failure), with `display.show_window: false`.
- **Linux**: use a systemd unit with `Restart=on-failure`.

## Extending to multiple cameras/stations

Run one process per camera, each with its own `config.yaml` (different
`station.id`, `CAMERA_URL`) pointing at the same shared database — all
events land in the same `dbo.StatusEvents` table, distinguished by
`StationID`.
