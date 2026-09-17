"""
Combined PPE + Container-OCR Field Dashboard.

One process (NiceGUI on FastAPI/uvicorn), meant to run directly on the Pi.
A single Start/Stop pair drives combined_live_pi.py as a background
subprocess -- that one script already does everything on the Hailo NPU in
one pass: detects people/helmets/container-ID crops, tracks + re-identifies
people, extracts + validates container IDs live, and records an annotated
video with running counts and confirmed IDs burned into the frames.

Start  -> launches combined_live_pi.py
Stop   -> sends SIGINT (same as Ctrl+C), waits for it to finalize the
          recording + both CSVs, then immediately pushes all three to the
          browser as ONE zip download ("released to host device") -- no
          extra click needed, though the button stays available to
          re-download later too.

No SSH/SCP needed for any of this -- it all rides the same browser
connection the dashboard page is already open on.
"""

import atexit
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path

from fastapi.responses import StreamingResponse
from nicegui import app, ui

PKG_DIR = Path(__file__).resolve().parent
COMBINED_SCRIPT = PKG_DIR / "combined_live_pi.py"

OUTPUT_DIR = PKG_DIR / "session_output"
RECORDING_PATH = OUTPUT_DIR / "annotated_recording.mp4"
CONTAINER_CSV_PATH = OUTPUT_DIR / "container_ids.csv"
NO_HELMET_CSV_PATH = OUTPUT_DIR / "no_helmet_report.csv"
RESULTS_ZIP_PATH = OUTPUT_DIR / "results.zip"

LIVE_FRAME_PATH = PKG_DIR / "live_frame.jpg"


# ============================================================== #
# Process control
# ============================================================== #
class StageRunner:
    STATS_RE = re.compile(
        r"frame=(\d+) \| FPS~([\d.]+) \| people=(\d+) \| no_helmet=(\d+) \| containers=(\d+)"
    )

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.log: deque[str] = deque()
        self.total_lines = 0
        self.stats: dict = {}

    def start(self, script: Path):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=PKG_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # own process group, so SIGINT targets it, not us
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        proc = self.proc
        for line in proc.stdout:
            self.log.append(line.rstrip("\n"))
            self.total_lines += 1
            m = self.STATS_RE.search(line)
            if m:
                self.stats = {
                    "frame": int(m.group(1)),
                    "fps": float(m.group(2)),
                    "people": int(m.group(3)),
                    "no_helmet": int(m.group(4)),
                    "containers": int(m.group(5)),
                }
        proc.wait()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop_gracefully(self, timeout=15):
        """Send Ctrl+C, let the script release the video writer and write
        both CSVs, escalate if it hangs."""
        if not self.running:
            return
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGINT)
        try:
            self.proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)

    def wait_for_marker(self, marker, force_kill_after=5):
        """Block until either the process exits OR `marker` shows up in the
        log. Some Hailo NPU driver threads linger past the script's own
        completion print, so a plain proc.wait() can hang -- give it a short
        grace period after the marker, then escalate."""
        while self.proc.poll() is None:
            if any(marker in line for line in self.log):
                break
            time.sleep(0.3)
        else:
            return

        try:
            self.proc.wait(timeout=force_kill_after)
            return
        except subprocess.TimeoutExpired:
            pass
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)


runner = StageRunner()
atexit.register(lambda: runner.stop_gracefully(timeout=5))


def _handle_sigterm(signum, frame):
    """`pkill -f dashboard.py` sends plain SIGTERM, which skips atexit --
    catch it explicitly so a running subprocess is always stopped cleanly
    instead of orphaned holding the NPU."""
    runner.stop_gracefully(timeout=5)
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)


# ============================================================== #
# State machine: idle -> running -> stopping -> done
# ============================================================== #
state = {"value": "idle"}
completed_runs = {"n": 0}


@app.get("/api/status")
def api_status():
    return {"state": state["value"], "completed_runs": completed_runs["n"]}


def start_session():
    if state["value"] not in ("idle", "done"):
        return
    state["value"] = "running"
    runner.log.clear()
    runner.total_lines = 0
    runner.stats = {}
    if LIVE_FRAME_PATH.exists():
        LIVE_FRAME_PATH.unlink()
    runner.start(COMBINED_SCRIPT)


def stop_session():
    if state["value"] != "running":
        return
    state["value"] = "stopping"

    def worker():
        runner.stop_gracefully()
        runner.wait_for_marker("SESSION_DONE")
        state["value"] = "done"
        completed_runs["n"] += 1
        push_downloads_pending["value"] = True

    threading.Thread(target=worker, daemon=True).start()


push_downloads_pending = {"value": False}


def release_results():
    """Push recording + both CSVs to the browser as ONE download -- works
    over the same connection the dashboard page is already open on, no
    SSH/SCP.

    Used to call ui.download() once per file (three calls back-to-back).
    Browsers commonly throttle/block multiple programmatic downloads fired
    without a fresh user gesture between each one -- confirmed live: on
    Safari/macOS only the LAST of the three (no_helmet_report.csv, last in
    the loop) consistently arrived, the recording and container CSV were
    silently dropped. Chrome/Windows was lenient enough that this went
    unnoticed there. Zipping everything into one archive and calling
    ui.download() exactly once sidesteps the multi-download restriction
    entirely instead of trying to out-time it (a small delay between calls
    would be a race against whatever each browser's actual policy is)."""
    present = [p for p in (RECORDING_PATH, CONTAINER_CSV_PATH, NO_HELMET_CSV_PATH) if p.exists()]
    if not present:
        ui.notify("No results yet -- run a session first.", type="warning")
        return
    with zipfile.ZipFile(RESULTS_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in present:
            zf.write(path, arcname=path.name)
    ui.download(RESULTS_ZIP_PATH, filename="usm_industrial_results.zip")


# ============================================================== #
# Live camera MJPEG stream -- tails LIVE_FRAME_PATH, written by
# combined_live_pi.py on every processed frame
# ============================================================== #
def _mjpeg_frames():
    boundary = b"--frame\r\n"
    while True:
        if LIVE_FRAME_PATH.exists():
            try:
                jpg = LIVE_FRAME_PATH.read_bytes()
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            except OSError:
                pass
        time.sleep(0.05)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


STATUS_STYLE = {
    "idle": ("IDLE", "#6b7280"),
    "running": ("RUNNING", "#22c55e"),
    "stopping": ("STOPPING", "#f59e0b"),
    "done": ("DONE", "#14b8a6"),
}


# ============================================================== #
# UI
# ============================================================== #
@ui.page("/")
def dashboard_page():
    ui.dark_mode(True)
    ui.colors(primary="#2563eb", accent="#f59e0b", positive="#22c55e", negative="#ef4444")
    ui.query("body").style("background:#0b0f14")

    with ui.header().classes("items-center justify-between").style(
        "background:#0f1620;border-bottom:1px solid #23303f"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.icon("radar", size="28px").classes("text-amber-400")
            ui.label("USM INDUSTRIAL").classes("text-xl font-bold tracking-widest")
            ui.label("Combined PPE + Container-OCR Dashboard").classes("text-sm text-gray-400")
        status_chip = ui.chip("IDLE", color="#6b7280").classes("text-white font-semibold px-4")

    with ui.row().classes("w-full p-4 gap-4 no-wrap items-start").style(
        "max-width:1500px;margin:auto"
    ):
        # ---------------- left column: camera ---------------- #
        with ui.column().classes("gap-4").style("flex:2;min-width:0"):
            with ui.card().classes("w-full").style(
                "background:#111823;border:1px solid #23303f"
            ):
                ui.label("Live Camera Feed").classes("text-lg font-semibold mb-2")
                video_holder = ui.html(
                    '<div style="color:#666;padding:3rem;text-align:center">'
                    "Camera idle — press Start</div>"
                ).classes("w-full rounded-lg overflow-hidden").style(
                    "background:#000;min-height:400px;display:flex;"
                    "align-items:center;justify-content:center"
                )
                stats_label = ui.label("").classes("text-sm text-gray-400 mt-2")

            with ui.card().classes("w-full").style(
                "background:#111823;border:1px solid #23303f"
            ):
                ui.label("Console").classes("text-lg font-semibold mb-2")
                log_view = ui.log(max_lines=None).classes("w-full font-mono text-xs").style(
                    "height:16rem;background:#000;color:#4ade80"
                )

        # ---------------- right column: controls + results ---------------- #
        with ui.column().classes("gap-4").style("flex:1;min-width:340px"):
            with ui.card().classes("w-full").style(
                "background:#111823;border:1px solid #23303f"
            ):
                ui.label("Mission Control").classes("text-lg font-semibold mb-2")
                with ui.row().classes("w-full gap-2"):
                    start_btn = ui.button(
                        "Start", icon="play_arrow", on_click=start_session
                    ).props("color=positive")
                    stop_btn = ui.button(
                        "Stop", icon="stop", on_click=stop_session
                    ).props("color=negative")
                    stop_btn.set_enabled(False)
                ui.label(
                    "Start records video + extracts container IDs + counts "
                    "no-helmet people, all at once. Stop finalizes and "
                    "sends the video + both CSVs to this browser as one zip."
                ).classes("text-xs text-gray-500 mt-2")

            with ui.card().classes("w-full").style(
                "background:#111823;border:1px solid #23303f"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Live Counts").classes("text-lg font-semibold")
                with ui.row().classes("w-full gap-4 mt-2"):
                    people_badge = ui.badge("0").classes("text-base")
                    ui.label("people")
                    no_helmet_badge = ui.badge("0").props("color=negative").classes("text-base")
                    ui.label("no-helmet")
                    containers_badge = ui.badge("0").props("color=primary").classes("text-base")
                    ui.label("containers ID'd")

            with ui.card().classes("w-full").style(
                "background:#111823;border:1px solid #23303f"
            ):
                ui.label("Results").classes("text-lg font-semibold mb-2")
                ui.button(
                    "Download results (zip)", icon="download", on_click=release_results
                ).props("color=primary").classes("w-full")

    log_cursor = {"n": 0}

    def refresh():
        text, color = STATUS_STYLE.get(state["value"], STATUS_STYLE["idle"])
        status_chip.set_text(text)
        status_chip.set_background_color(color)

        start_btn.set_enabled(state["value"] in ("idle", "done"))
        stop_btn.set_enabled(state["value"] == "running")

        if state["value"] in ("running", "stopping"):
            video_holder.set_content(
                '<img src="/video_feed" style="width:100%;display:block" />'
            )
        elif state["value"] == "idle":
            video_holder.set_content(
                '<div style="color:#666;padding:3rem;text-align:center">'
                "Camera idle — press Start</div>"
            )
        else:
            video_holder.set_content(
                '<div style="color:#666;padding:3rem;text-align:center">'
                "Session stopped</div>"
            )

        st = runner.stats
        stats_label.set_text(
            f"frame {st['frame']} | {st['fps']:.1f} FPS" if st else ""
        )
        people_badge.set_text(str(st.get("people", 0)))
        no_helmet_badge.set_text(str(st.get("no_helmet", 0)))
        containers_badge.set_text(str(st.get("containers", 0)))

        lines = list(runner.log)
        oldest_index = runner.total_lines - len(lines)
        start_offset = max(0, log_cursor["n"] - oldest_index)
        for line in lines[start_offset:]:
            log_view.push(line)
        log_cursor["n"] = runner.total_lines

        if push_downloads_pending["value"]:
            push_downloads_pending["value"] = False
            release_results()

    ui.timer(1.0, refresh)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=8000,
        title="USM Industrial Combined Dashboard",
        dark=True,
        reload=False,
        show=False,
    )
