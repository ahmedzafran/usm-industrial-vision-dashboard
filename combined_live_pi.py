#!/usr/bin/env python3
"""
Fully Hailo-accelerated live pipeline for the Pi: TWO 2-class detectors --
16mm_toy_container.hef (blurry_ocr_image, sharp_ocr_image) for container-ID
crop detection, and yolo26s_person_helmet.hef (helmet, person) for PPE
detection -- run as two separate NPU inference passes per frame. OCR
recognition (PaddleOCR det+rec) and person identity confirmation
(repvgg_a0_person_reid_512) also run on the Hailo NPU.

This replaces an earlier single-combined-4-class-model version. That model
required real ONNX graph surgery to get on-chip NMS at all (see git history
if curious), and even once NMS was confirmed structurally correct in the
compiled graph, live testing on the Pi showed it detecting essentially
nothing on clearly legible content -- calibration-only quantization
(Adaround was blocked by a GPU driver issue on the dev machine) stacked on
top of the custom NMS-config workaround was evidently too much combined
degradation to trust. Two separate 2-class models sidestep the root
technical issue entirely: Hailo's shape-based box-vs-classification
detector fails specifically when a model's class count numerically matches
its box-regression channel count (4, for this reg_max=1 architecture) --
2 classes never collides with 4, so both hefs compiled with clean,
un-hacked auto-detected on-chip NMS on the first try, same proven recipe
as the original working 16mm_toy_container.hef.

Meant to be launched by dashboard.py as a subprocess, but runs standalone
too: records live camera video with boxes, running counts, and confirmed
container IDs burned into the frames AS THEY'RE CONFIRMED (not after the
fact), and on Ctrl+C (dashboard sends SIGINT the same way) finalizes the
recording plus two CSV reports.

Design notes carried over from where each piece was proven:
  - CORRECTION: on-chip NMS was assumed here but isn't actually what got
    compiled -- `hailortcli parse-hef` on 16mm_toy_container.hef shows 6
    raw outputs (3 box heads + 3 class heads across strides 8/16/32), same
    surprise package_ocr_usm_hailo/hailo_pipeline_live_pi.py hit for this
    exact file. HailoBoxDetector below now detects and decodes this itself
    (manual anchor-free decode + per-class NMS) so it still hands back the
    on-chip-NMS-shaped per-class [x1,y1,x2,y2,score] list (normalized 0..1
    vs the letterboxed 640x640 input) that the rest of this file expects --
    but don't assume a newly (re)compiled hef here has real on-chip NMS
    without checking `hailortcli parse-hef` first.
  - TWO independent SORT trackers (person objects vs. OCR-class objects) --
    combined_workflow.py's finding: these move independently and sharing one
    tracker's identity space corrupts both.
  - Person identity: local SORT track id -> stable global id via
    reid_engine.py's GlobalIdentityTracker + the real Hailo ReID model, so a
    worker who steps out of frame and back in isn't double-counted. Runs the
    Hailo embed call only for genuinely new local tracks or periodic
    refreshes (see reid_engine.py), not literally every detection.
  - Container ID: live incremental finalization (combined_workflow.py's
    approach, not hailo_pipeline_live_pi.py's finalize-on-track-drop) --
    the instant a track's accumulated candidates clear pick_majority_id()'s
    bar, the ID is locked in and drawn on every subsequent frame showing
    that track, which is what "confirmed ID visible in the video in real
    time" actually requires.
  - Only sharp_ocr_image crops feed OCR/voting (not blurry) -- this runs on
    the Pi's own native training-domain camera, where
    hailo_pipeline_live_pi.py already validated that the sharp/blurry split
    is reliable enough to gate on.

Latency optimizations (dev-machine-verifiable ones only -- no Hailo hardware
available there to profile anything deeper; these are all either pure CPU
math with no behavior change, or synthetic-tested):
  - Both detectors are compiled at the same 640x640 input size, so the
    letterboxed frame is byte-identical for either -- computed once per
    frame and reused, instead of a second full-frame resize (falls back to
    computing separately if a future recompile changes one model's size).
  - default_preprocess's letterbox resize uses INTER_AREA instead of
    INTER_CUBIC -- every call site here is a downscale (full camera frame,
    or an already-12x-upscaled OCR crop, both shrinking down to a model's
    fixed input size), and INTER_AREA is simultaneously the
    correct-quality choice for shrinking *and* cheaper to compute than
    CUBIC -- not a quality/speed tradeoff, a straight win.
  - The manual raw-dense decode path's grid-center coordinates (cx, cy)
    depend only on each output's (h, w), which is fixed once a hef is
    compiled -- precomputed once in HailoBoxDetector.__init__ instead of
    rebuilt via np.meshgrid on every single frame. Verified against a
    synthetic known-detection test that the decoded box math is unchanged.
  - live_frame.jpg (the dashboard's MJPEG preview) is now written every
    3rd frame (~10fps at a 30fps camera) instead of every frame --
    dashboard.py's own poll is far coarser than 30fps, so this was pure
    wasted JPEG-encode + disk-write competing with the actual inference
    loop for CPU time.
  Bigger, riskier wins deliberately NOT attempted here without hardware to
  verify against: switching the two detector .infer() calls (and/or the
  OCR det+rec calls) to HailoRT's async run_async()/callback API instead
  of the current synchronous blocking .run() so the NPU and CPU postprocess
  work overlap instead of serializing; and reducing person/helmet detection
  to every-2nd-frame (SORT's Kalman filter already tolerates gaps) as an
  accuracy/speed tradeoff only you can call given the hardware in front of
  you.

Requires (Pi side): hailo_platform (apt, not pip), opencv-python, numpy,
scipy, shapely, pyclipper. See requirements.txt.
"""

import csv
import os
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
import pyclipper
import re

from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm

from reid_engine import (
    ReIDEngine, GlobalIdentityTracker, crop_person_px, crop_person_for_reid_px, REID_HEF_DEFAULT,
)


# ------------------------------- Config ----------------------------------- #
# Camera settings match the OCR pipeline's validated capture (fixed shutter,
# fast enough to freeze handheld text, ISP tuned for fine text) -- this is
# one physical camera feeding two models, so one capture profile has to
# serve both jobs. If live testing shows person/helmet recall suffers (or
# the feed looks miscolored) at these settings, this is the first place to
# revisit -- these values were tuned specifically for close-up OCR text, not
# a general person/helmet scene.
CAM_W, CAM_H, CAM_FPS = 2028, 1520, 30
SHUTTER_US = 3000
GAIN = 6.0
DENOISE_MODE = "cdn_hq"
SHARPNESS = 1.5
MJPEG_QUALITY = 95

CONTAINER_DET_HEF_PATH = "16mm_toy_container.hef"
PERSON_HELMET_HEF_PATH = "yolo26s_person_helmet.hef"
OCR_DET_HEF_PATH = "paddleocr_det_h10.hef"
OCR_REC_HEF_PATH = "paddleocr_rec_v5_h10.hef"
OCR_REC_DICT_PATH = "ppocrv5_dict.txt"

# Class order read from each model's own exported onnx metadata:
#   python3 -c "import onnx; m=onnx.load('X.onnx'); \
#     print([p.value for p in m.metadata_props if p.key=='names'])"
BLURRY_IDX, SHARP_IDX = 0, 1
CONTAINER_CLASS_NAMES = {BLURRY_IDX: "blurry_ocr_image", SHARP_IDX: "sharp_ocr_image"}

HELMET_IDX, PERSON_IDX = 0, 1
PERSON_HELMET_CLASS_NAMES = {HELMET_IDX: "helmet", PERSON_IDX: "person"}

DET_CONF_THRESHOLD = 0.25

# ---- Local (per-session) trackers: motion-only SORT, one per object kind ----
# SORT already runs a (constant-velocity) Kalman filter per track
# (KalmanBoxTracker below) to predict box position across frames -- what
# actually breaks under erratic movement is that constant-velocity
# assumption itself (a fast head turn isn't constant velocity) and how
# strict IOU-based re-association is, not the absence of a filter. Loosened
# both a bit so a track survives erratic motion instead of being dropped
# and re-created as a new local id (which is what forces GlobalIdentityTracker
# to re-run its whole re-id decision from scratch).
SORT_MAX_AGE = 45       # was 30 -- more frames of no matching detection tolerated before a track is dropped
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.2  # was 0.3 -- more forgiving of box jump between frames from fast/erratic movement

# ---- Person-box head/body dedup ----
# yolo26s_person_helmet.hef's training data boxed BOTH a head-scale region
# and a full-body region as class "person" for the same physical target --
# left unfiltered this mints two SORT tracks (and two ReID gallery entries)
# per real person. We keep only the head/face-scale "person" box per target
# (filter_head_scale_person_boxes below) as the canonical detection for
# tracking/counting/helmet-matching, but crop the paired body box (also
# returned by that function) for the actual ReID embedding -- see its
# docstring for why.
PERSON_HEAD_ASPECT_MIN = 0.5      # height/width below this is wider than tall -- not a head
PERSON_HEAD_ASPECT_MAX = 1.8      # height/width above this is a standing body, not a head
PERSON_HEAD_MAX_FRAME_FRAC = 0.5  # a "head" taller than this fraction of the frame is a bad read
PERSON_CONTAINMENT_SUPPRESS = 0.6 # a box >=60% contained inside a bigger same-class box is that bigger box's duplicate body region -- drop the bigger one, keep the smaller (head)
# A motion-blurred/fast-moving target can make the detector fire two
# distinct, non-nested head-scale boxes a beat apart for the same physical
# person -- containment alone (needs near-total nesting) misses this, so
# also suppress a plain-IoU duplicate at this threshold. Real adjacent
# people's heads essentially never overlap this much.
PERSON_OVERLAP_SUPPRESS = 0.35

# ---- Helmet/person association ----
# person_boxes are head-scale (see above), so the box itself IS the head
# region -- pad the matching zone upward by this fraction of the head box's
# height before checking helmet overlap, since a helmet's crown often sits
# just above a tight head detection.
HEAD_ZONE_UPWARD_PAD = 0.3
MIN_HEAD_OVERLAP = 0.3
MIN_OBSERVATIONS = 3  # ignore a global id's helmet vote until seen this many times

# ---- Hailo person ReID (reid_engine.py) ----
REID_HEF_PATH = REID_HEF_DEFAULT  # /usr/local/hailo/resources/models/hailo10h/repvgg_a0_person_reid_512.hef
# Every value tried before 2026-08-19 (0.50 -> 0.35 -> 0.15) was tuned while
# repvgg_a0_person_reid_512.hef was silently missing from this Pi -- see
# reid_engine.py's ReIDEngine docstring / the README's setup note. reid.ok
# was False the entire time, so resolve() was hitting its no-signal fallback
# (instant new id, no threshold ever consulted) regardless of this number;
# none of that tuning ever actually ran.
#
# 0.40 (the first real estimate, from a single-person clip with no
# different-person data to compare against) turned out to still be too low:
# a real two-person session (REID_DUMP_PATH analysis, 2026-08-19) showed
# person A re-matching itself at 0.87-0.96 across track churns (correct,
# and exactly the re-entry stability this project wants), but person B --
# a visually different second person confirmed by eye at two separate
# frames -- scored 0.50-0.78 against person A's gallery and got silently
# folded into the same global id instead of minting a new one the whole
# rest of that session. There's a real, clean gap between those two
# clusters in this data (same-person floor ~0.87, different-person ceiling
# ~0.78) -- 0.40 sat below both, rejecting nothing. Set to 0.82: inside
# that gap, biased toward the different-person side for safety margin.
# NOTE: an earlier, less representative single-person test (not the real
# pipeline's actual crop path) saw same-person scores as low as 0.54 --
# if 0.82 turns out to split a real re-entry into a new id, that's the
# next data point to reconcile, not a reason to silently revert this.
REID_COSINE_THRESHOLD = 0.82
REID_MAX_SAMPLES = 8
REID_UPDATE_INTERVAL = 12
REID_TRACK_TTL = 300
REID_MIN_CROP = 24
# A brand-new local track doesn't get a fresh global id on its first failed
# gallery match -- one bad crop (blur, odd pose, bad face angle) shouldn't be
# enough evidence to decide "this is a new person" when it might really be a
# re-entry, and this is exactly what makes leaving-and-re-entering-frame
# work: SORT itself usually can't reuse the same local track id across a
# real departure, so this confirm window, checked against the session's
# persistent gallery, is what's actually responsible for recognizing that
# new local track as the same returning person. Literal attempt count, not
# a time budget -- the real-world wait will vary with however fast this
# Pi's shared-NPU pipeline happens to be running (observed ~6.9-7fps).
#
# NOTE ON DESIGN HISTORY: a simpler one-shot version (decide once,
# immediately, on the first crop clearing MIN_CROP_SHARPNESS, no
# accumulation) was tried, since that's both what this codebase was
# originally ported from and what an independent design review recommended.
# It was tested live on this actual rig and measurably performed WORSE, not
# better -- do not re-simplify back to one-shot without new evidence it
# would help on THIS deployment.
#
# 60 was also the direct cause of a separate, real problem: real-session
# data (REID_DUMP_PATH, 2026-08-19) shows the first-ever person in a
# session didn't get counted until frame ~90 (~13-14s at this rig's
# observed ~6.5fps) -- someone who walks through in less time than that
# would leave without ever being counted at all. This number ONLY governs
# how long resolve() waits before concluding "no match, this is a new
# person" -- it has NO effect on re-entry matching, which is evaluated
# fresh on every single attempt regardless of window size (a re-entry can
# resolve on attempt 1 of a brand-new confirm window just as easily as
# attempt 60). Lowering it speeds up counting genuinely new/transient
# people without weakening the re-entry stability this project wants to
# keep. Lowered from 60 to 20 (~3-4s at this rig's fps) -- still real
# patience against a single bad crop, just not 13+ seconds of it.
REID_NEW_ID_CONFIRM_ATTEMPTS = 20
# The sharpness gate rejects a ReID crop before it's ever embedded, spent as
# a confirm attempt, or added to the gallery. Blur was observed live to be a
# major driver of false "new person" mints: a blurry crop still embeds and
# compares, it just produces a near-meaningless embedding that reliably
# fails to match -- and if the confirm window ran out while every crop seen
# was blurry, that noisy embedding would become the new person's ONLY
# gallery entry, making their own later (possibly clean) re-entries harder
# to match too.
#
# ADAPTIVE by default now, not a fixed number: real crop sharpness swings
# wildly with scene/crop framing/lighting -- measured 7 to 830 on the SAME
# test rig depending on exactly what fell inside the crop -- so any single
# fixed threshold is wrong somewhere, and a deployment site's real lighting
# may be worse than whatever this was last tuned against. The effective
# threshold is instead the REID_SHARPNESS_PERCENTILE-th percentile of every
# sharpness score actually observed this session (a rolling window, see
# REID_SHARPNESS_HISTORY_SIZE) -- lenient by default (15th percentile:
# reject only the blurriest ~15% of what's typical here), and it follows
# whatever this session's real conditions turn out to be instead of
# guessing them in advance. The gate stays wide open (accepts everything)
# until REID_SHARPNESS_WARMUP_SAMPLES readings exist to calibrate against.
#
# MIN_CROP_SHARPNESS is now just a hard FLOOR underneath the adaptive
# threshold (never goes below this even if the adaptive percentile would),
# so a scene where EVERYTHING is degenerate (camera fault, solid color,
# total wash-out) still gets filtered rather than the adaptive threshold
# chasing it down to near-zero. Kept low/lenient on purpose. Set
# REID_ADAPTIVE_SHARPNESS = False to go back to a pure fixed threshold at
# MIN_CROP_SHARPNESS if the adaptive behavior ever needs to be ruled out
# during debugging.
REID_ADAPTIVE_SHARPNESS = True
REID_SHARPNESS_PERCENTILE = 15
REID_SHARPNESS_HISTORY_SIZE = 200
REID_SHARPNESS_WARMUP_SAMPLES = 30
MIN_CROP_SHARPNESS = 5.0
# Safety valve for the sharpness gate (fixed or adaptive): if a
# still-unresolved local track goes this many frames without EVER producing
# a single crop that clears the gate, the gate gives up and starts
# accepting whatever crop is available for that track from then on, rather
# than leaving them stuck showing "ID?" and never counted for their entire
# time in frame. Confirmed live on 2026-08-19: a ~28s session (184 frames)
# with one continuously-visible person in dim/soft lighting, zero crops
# ever cleared the (then-fixed) sharpness gate, and people=0 for the entire
# session -- this was a real deadlock, not a hypothetical one. Now mostly a
# backstop underneath the adaptive gate rather than the normal path, but
# kept in place since adaptive calibration can still misjudge a scene that
# changes drastically partway through (e.g. lights physically dim mid-
# session). ~90 frames (~13-15s at this rig's observed ~6.5fps) gives the
# gate a real chance to work under normal lighting before giving up.
REID_BLUR_GATE_TIMEOUT_FRAMES = 90
# If set, every embedding computed this session gets appended as
# (frame_idx, local_track_id, embedding) to this pickle file (rewritten
# periodically, not just at the end, so a mid-session stop doesn't lose it)
# -- purely for OFFLINE analysis: comparing cosine scores between crops that
# came from the SAME local track (should cluster high) vs. different tracks
# (should cluster low) to find the actual optimal REID_COSINE_THRESHOLD for
# this camera/model/population instead of guessing it from live anecdotes.
# None disables collection entirely.
REID_DUMP_PATH = None  # set below, once OUTPUT_DIR exists
# Real-time matching (above) gets exactly one noisy shot at a decision per
# attempt. This is a second, offline-style pass over the WHOLE gallery, run
# every this-many frames (cheap to run rarely -- it's O(distinct people^2)):
# any two "different" identities whose accumulated samples cross-match well
# are almost certainly one real person who got split into two ids by a
# failed live match, and get folded back into one. Uses REID_COSINE_THRESHOLD
# as its merge threshold too (GlobalIdentityTracker defaults to that when no
# override is given) -- watch the "[ReID] gallery self-check: merged ..."
# console line: if it fires for two people you know are actually different,
# raise REID_COSINE_THRESHOLD (this reuses it) rather than adding a separate
# merge threshold.
REID_GALLERY_MERGE_CHECK_INTERVAL = 30

# ---- OCR crop capture + ID normalization ----
CROP_PADDING = 5
MIN_CROP_W = 10
MIN_CROP_H = 10
UPSCALE_FACTOR = 12
APPLY_SHARPEN = True
DET_BIN_THRESH = 0.3
DET_BOX_THRESH = 0.6
DET_UNCLIP_RATIO = 1.5
NEAR_DUPLICATE_MAX_DISTANCE = 2

# normalize_unique_id_candidates adds this to a candidate's score when its
# trailing digit matches the ISO 6346 check digit computed from the rest of
# the id -- a real structural validation (mod-11 checksum), not just an OCR
# confidence heuristic. letter_bonus alone tops out at 10 and gap is always
# >= 0, so a score >= CHECK_DIGIT_BONUS can only happen when this bonus
# fired -- used below to detect "this exact read's check digit validated."
CHECK_DIGIT_BONUS = 20

# A track's id is locked in once its winning near-duplicate cluster (see
# pick_majority_id) has enough corroborating crop reads -- without this, a
# single OCR misread on the very first sharp crop was enough to finalize
# immediately (pick_majority_id returns a non-None winner as soon as
# there's 1 candidate).
#
# Two tiers: if any corroborating read for the winning id actually passed
# ISO 6346 check-digit validation, that's a much stronger signal than mere
# repetition (a wrong OCR read matching the checksum by chance is unlikely),
# so it needs far fewer votes to finalize -- extra votes past that point
# just add latency without adding real confidence. Reads that never
# validate (e.g. the check-digit region wasn't legible) fall back to
# requiring more corroboration through repetition alone.
MIN_CONTAINER_VOTES = 3              # required when no read validated its check digit
MIN_CONTAINER_VOTES_VALIDATED = 1    # required once a read's check digit has validated

# ---- Output (fixed names -- dashboard.py clears this dir before each run) ----
OUTPUT_DIR = "session_output"
RECORDING_PATH = os.path.join(OUTPUT_DIR, "annotated_recording.mp4")
CONTAINER_CSV_PATH = os.path.join(OUTPUT_DIR, "container_ids.csv")
NO_HELMET_CSV_PATH = os.path.join(OUTPUT_DIR, "no_helmet_report.csv")
REID_DUMP_PATH = os.path.join(OUTPUT_DIR, "reid_embeddings_dump.pkl")  # see REID_DUMP_PATH comment above

LIVE_FRAME_PATH = "live_frame.jpg"
LIVE_FRAME_MAX_WIDTH = 960
LIVE_FRAME_QUALITY = 80
# dashboard.py's own browser-side poll is 0.05s and its stats panel only
# refreshes once/sec -- writing a fresh JPEG every single processed frame
# (up to 30x/sec) spends CPU/disk-IO the viewer can't even use. Every 3rd
# frame (~10fps preview at a 30fps camera) is still smooth to watch and
# frees that budget for the actual inference pipeline.
LIVE_FRAME_EVERY_N_FRAMES = 3

LOG_EVERY_SEC = 5.0

# ---- Drawing ----
COLOR_PERSON_OK = (0, 200, 0)
COLOR_PERSON_NO_HELMET = (0, 0, 255)
COLOR_PERSON_PENDING = (160, 160, 160)  # still confirming identity -- not counted yet (see REID_NEW_ID_CONFIRM_ATTEMPTS)
COLOR_HELMET = (0, 220, 220)
COLOR_BLURRY = (0, 140, 255)
COLOR_SHARP = (255, 200, 0)
COLOR_ID_CONFIRMED = (0, 255, 0)
# --------------------------------------------------------------------------- #


# =====================================================================
# Camera ingest: spawn rpicam-vid as an MJPEG stream, decode BGR frames
# =====================================================================
def start_camera(width, height, fps):
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--codec", "mjpeg",
        "--inline",
        "--width", str(width),
        "--height", str(height),
        "--framerate", str(fps),
        "--shutter", str(SHUTTER_US),
        "--gain", str(GAIN),
        "--denoise", DENOISE_MODE,
        "--sharpness", str(SHARPNESS),
        "--quality", str(MJPEG_QUALITY),
        "-o", "-",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
    )


def mjpeg_frames(proc, buf=4096):
    data = b""
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    while True:
        chunk = proc.stdout.read(buf)
        if not chunk:
            return
        data += chunk
        while True:
            a = data.find(SOI)
            b = data.find(EOI)
            if a != -1 and b != -1 and b > a:
                jpg = data[a:b + 2]
                data = data[b + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
            else:
                break


# =====================================================================
# Self-contained SORT (Bewley et al., 2016), numpy + scipy only --
# instantiated twice below (person tracker, OCR-class tracker)
# =====================================================================
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    s = max(float(x[2, 0]), 1e-6)
    r = max(float(x[3, 0]), 1e-6)
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.0
    cx, cy = float(x[0, 0]), float(x[1, 0])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def iou_batch(dets, trks):
    if len(dets) == 0 or len(trks) == 0:
        return np.zeros((len(dets), len(trks)), dtype=float)
    d = np.expand_dims(dets[:, :4], 1)
    t = np.expand_dims(trks[:, :4], 0)
    xx1 = np.maximum(d[..., 0], t[..., 0])
    yy1 = np.maximum(d[..., 1], t[..., 1])
    xx2 = np.minimum(d[..., 2], t[..., 2])
    yy2 = np.minimum(d[..., 3], t[..., 3])
    w = np.clip(xx2 - xx1, 0.0, None)
    h = np.clip(yy2 - yy1, 0.0, None)
    inter = w * h
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    union = area_d + area_t - inter
    return np.where(union > 0, inter / union, 0.0)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = convert_bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def predict(self):
        if float(self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6] *= 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.x)

    def get_state(self):
        return convert_x_to_bbox(self.x)


def associate(dets, trks, iou_threshold):
    if len(trks) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(dets)),
                np.empty((0,), dtype=int))

    iou_matrix = iou_batch(dets, trks)
    if min(iou_matrix.shape) > 0:
        row, col = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row, col)))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [d for d in range(len(dets))
                      if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trks))
                      if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    matches = np.concatenate(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t in range(len(trks)):
            pos = self.trackers[t].predict()
            trks[t, :4] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        valid = ~np.any(np.isnan(trks), axis=1)
        trks = trks[valid]
        for t in reversed(to_del):
            self.trackers.pop(t)

        if len(dets) == 0:
            dets = np.empty((0, 5))
        matched, unmatched_dets, _ = associate(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if ret else np.empty((0, 5))


# =====================================================================
# Helmet/person association
# =====================================================================
def box_overlap_fraction(box, other):
    ax1, ay1, ax2, ay2 = box
    bx1, by1, bx2, by2 = other
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    b_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / b_area


def match_helmets_to_persons(person_boxes, helmet_boxes):
    n_p, n_h = len(person_boxes), len(helmet_boxes)
    has_helmet = [False] * n_p
    if n_p == 0 or n_h == 0:
        return has_helmet

    candidates = []
    for pi, pbox in enumerate(person_boxes):
        px1, py1, px2, py2 = pbox
        head_zone = (px1, py1 - HEAD_ZONE_UPWARD_PAD * (py2 - py1), px2, py2)
        for hi, hbox in enumerate(helmet_boxes):
            overlap = box_overlap_fraction(head_zone, hbox)
            if overlap >= MIN_HEAD_OVERLAP:
                candidates.append((overlap, pi, hi))

    candidates.sort(reverse=True, key=lambda c: c[0])
    used_helmets = set()
    for overlap, pi, hi in candidates:
        if has_helmet[pi] or hi in used_helmets:
            continue
        has_helmet[pi] = True
        used_helmets.add(hi)

    return has_helmet


def _box_wh(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _containment_fraction(inner, outer):
    """Fraction of `inner`'s own area that overlaps `outer` -- unlike IoU,
    this stays close to 1.0 even when `outer` is much bigger than `inner`,
    which is exactly the head-box-nested-inside-body-box case plain IoU-based
    NMS misses (a small head box vs. a much larger body box has a low IoU
    even when the head sits entirely inside the body region)."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    ax1, ay1 = max(ix1, ox1), max(iy1, oy1)
    ax2, ay2 = min(ix2, ox2), min(iy2, oy2)
    iw, ih = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    inter = iw * ih
    inner_area = max(1e-6, (ix2 - ix1) * (iy2 - iy1))
    return inter / inner_area


def _iou(a, b):
    """Standard intersection-over-union -- catches the case containment
    misses: two SAME-scale head boxes for one motion-blurred target that are
    offset from each other (a smeared/moving subject can make the detector
    fire two distinct, non-nested boxes a beat apart) rather than one nested
    inside the other. Two genuinely different people's heads essentially
    never reach this much overlap unless they're face-to-face close, so this
    stays safe for real adjacent people."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_head_scale_person_boxes(person_boxes, frame_h):
    """Keeps only the 'person' detections that are head/face-scale, dropping
    the full-body duplicates the hef also emits for the same target (see the
    comment above PERSON_HEAD_ASPECT_MIN) from tracking/counting/helmet
    matching. Two passes:
      1. Aspect-ratio + absolute-size filter -- classifies each raw
         detection as head-like or not (a standalone body box with no
         paired head this frame is simply dropped).
      2. Containment suppression -- among head-like boxes, if one is mostly
         nested inside another they're almost certainly two head-like reads
         of the same target; keep the smaller.

    Returns (head_boxes, body_ref_boxes): head_boxes is the deduped list
    used for everything downstream (tracking, counting, helmet matching,
    drawing); body_ref_boxes is the same length, each entry either the raw
    full-body box the hef also emitted for that head this frame (the
    largest raw detection that contains it), or None if only the head fired.
    Callers should crop that real body box for the ReID embedding instead of
    the head box -- repvgg_a0_person_reid_512 was validated against
    whole-body crops, and repeated live testing on this rig confirmed a
    squashed face crop just doesn't carry enough signal for reliable
    re-entry recognition (works fine standing still in good focus, fails
    under blur or a fresh angle on return -- exactly what a network never
    trained on faces would do)."""
    head_like = []
    for box in person_boxes:
        w, h = _box_wh(box)
        if w <= 0 or h <= 0:
            continue
        aspect = h / w
        head_like_shape = PERSON_HEAD_ASPECT_MIN <= aspect <= PERSON_HEAD_ASPECT_MAX
        if head_like_shape and h <= PERSON_HEAD_MAX_FRAME_FRAC * frame_h:
            head_like.append(box)

    if len(head_like) >= 2:
        by_area = sorted(head_like, key=lambda b: _box_wh(b)[0] * _box_wh(b)[1])
        deduped = []
        for box in by_area:
            is_duplicate_head = any(
                _containment_fraction(kb, box) >= PERSON_CONTAINMENT_SUPPRESS
                or _iou(kb, box) >= PERSON_OVERLAP_SUPPRESS
                for kb in deduped
            )
            if not is_duplicate_head:
                deduped.append(box)
        head_like = deduped

    body_refs = []
    for head in head_like:
        head_area = _box_wh(head)[0] * _box_wh(head)[1]
        best, best_area = None, head_area
        for box in person_boxes:
            if box is head:
                continue
            area = _box_wh(box)[0] * _box_wh(box)[1]
            if area > best_area and _containment_fraction(head, box) >= PERSON_CONTAINMENT_SUPPRESS:
                best, best_area = box, area
        body_refs.append(best)

    return head_like, body_refs


# =====================================================================
# Letterbox helpers + Hailo box detector (on-chip NMS)
# =====================================================================
def default_preprocess(image, model_w, model_h):
    """Letterbox resize + pad. INTER_AREA, not INTER_CUBIC -- every call site
    here is a downscale (full camera frame -> 640x640 detector input, or an
    already-12x-upscaled OCR crop -> the rec/det model's fixed input size),
    and INTER_AREA is both the correct-quality choice for shrinking *and*
    measurably cheaper than CUBIC -- a straight win, not a quality/speed
    tradeoff, for every current caller."""
    img_h, img_w = image.shape[:2]
    scale = min(model_w / img_w, model_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded


def denormalize_letterbox_box(box_xyxy_norm, orig_w, orig_h, model_w, model_h):
    scale = min(model_w / orig_w, model_h / orig_h)
    new_w, new_h = orig_w * scale, orig_h * scale
    x_off = (model_w - new_w) / 2.0
    y_off = (model_h - new_h) / 2.0

    x1n, y1n, x2n, y2n = box_xyxy_norm
    x1 = (x1n * model_w - x_off) / scale
    y1 = (y1n * model_h - y_off) / scale
    x2 = (x2n * model_w - x_off) / scale
    y2 = (y2n * model_h - y_off) / scale
    return x1, y1, x2, y2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _decode_raw_dense_scales(scale_outputs, model_w, model_h,
                              conf_floor=0.05, nms_iou_threshold=0.5):
    """Manual decode for a raw anchor-free yolo26s head (reg_max=1, so box
    channels are already (l,t,r,b) distances, no DFL/softmax needed) --
    same math as package_ocr_usm_hailo/hailo_pipeline_live_pi.py, which hit
    the same surprise for this same 16mm_toy_container.hef (compiling
    cleanly does NOT guarantee on-chip NMS actually landed in the graph).

    Returns Hailo-on-chip-NMS-shaped output: a per-class list of
    [x1,y1,x2,y2,score] boxes normalized 0..1 against (model_w, model_h),
    so callers (decode_detections below) don't need to know which kind of
    .hef they're looking at.

    scale_outputs entries are (box_hwc, cls_hwc, stride, cx, cy) -- cx/cy are
    the grid-center coordinate arrays, precomputed ONCE at detector init
    (HailoBoxDetector._scale_pairs) rather than rebuilt via np.meshgrid on
    every single frame for a value that never changes for a given hef.
    """
    all_boxes, all_scores, all_classes, num_classes = [], [], [], 0
    for box_hwc, cls_hwc, stride, cx, cy in scale_outputs:
        num_classes = cls_hwc.shape[-1]
        l, t, r, b = box_hwc[..., 0], box_hwc[..., 1], box_hwc[..., 2], box_hwc[..., 3]
        x1 = (cx - l) * stride
        y1 = (cy - t) * stride
        x2 = (cx + r) * stride
        y2 = (cy + b) * stride
        boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
        scores = _sigmoid(cls_hwc.reshape(-1, num_classes))
        best_cls = np.argmax(scores, axis=-1)
        best_score = scores[np.arange(scores.shape[0]), best_cls]
        keep = best_score >= conf_floor
        if not np.any(keep):
            continue
        all_boxes.append(boxes[keep])
        all_scores.append(best_score[keep])
        all_classes.append(best_cls[keep])

    per_class = [[] for _ in range(num_classes)]
    if not all_boxes:
        return per_class

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    for class_id in np.unique(classes):
        mask = classes == class_id
        cls_boxes, cls_scores = boxes[mask], scores[mask]
        wh_boxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    for x1, y1, x2, y2 in cls_boxes]
        keep_idx = cv2.dnn.NMSBoxes(wh_boxes, cls_scores.tolist(), conf_floor, nms_iou_threshold)
        if len(keep_idx) == 0:
            continue
        for i in np.array(keep_idx).reshape(-1):
            x1, y1, x2, y2 = cls_boxes[i]
            per_class[int(class_id)].append([
                float(x1) / model_w, float(y1) / model_h,
                float(x2) / model_w, float(y2) / model_h,
                float(cls_scores[i]),
            ])
    return per_class


class HailoBoxDetector:
    """Hailo object detector (HailoRT 5.x InferModel API). Transparently
    handles two kinds of compiled .hef output:
      - Single output: genuine on-chip NMS -- used as-is (Hailo's own
        per-class [x1,y1,x2,y2,score] normalized-0..1 list format).
      - Multiple outputs: raw per-scale dense anchor-free head, no on-chip
        NMS despite compiling cleanly -- 16mm_toy_container.hef turned out
        to be exactly this (confirmed via `hailortcli parse-hef`: 6 outputs,
        3 box heads + 3 class heads across strides 8/16/32), contradicting
        the on-chip-NMS assumption this class originally shipped with.
        Decoded here into the same Hailo-NMS-shaped output so
        decode_detections() below never needs to know which kind it got."""

    def __init__(self, hef_path, timeout_ms=10000):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self._vdevice = VDevice(params)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        try:
            self._infer_model.input().set_format_type(FormatType.UINT8)
        except Exception:
            pass

        self._output_names = list(self._infer_model.output_names)
        for name in self._output_names:
            try:
                self._infer_model.output(name).set_format_type(FormatType.FLOAT32)
            except Exception:
                pass

        self._configured = self._infer_model.configure()
        self._bindings = self._configured.create_bindings()

        ishape = self._infer_model.input().shape
        self.in_h, self.in_w = int(ishape[0]), int(ishape[1])
        self._in_buf = np.empty(ishape, dtype=np.uint8)
        self._bindings.input().set_buffer(self._in_buf)

        self._out_bufs = {}
        for name in self._output_names:
            shape = tuple(int(d) for d in self._infer_model.output(name).shape)
            buf = np.empty(shape, dtype=np.float32)
            self._out_bufs[name] = buf
            self._bindings.output(name).set_buffer(buf)

        self._raw_dense = len(self._output_names) > 1
        if self._raw_dense:
            box_by_hw, cls_by_hw = {}, {}
            for name in self._output_names:
                h, w, c = self._out_bufs[name].shape
                (box_by_hw if c == 4 else cls_by_hw)[(h, w)] = name
            # (box_name, cls_name, stride, cx, cy) -- cx/cy (grid-center
            # coordinates) depend only on (h, w), which is fixed once the hef
            # is compiled, so they're computed once here instead of every
            # frame in _decode_raw_dense_scales.
            self._scale_pairs = []
            for (h, w), box_name in box_by_hw.items():
                cls_name = cls_by_hw.get((h, w))
                if cls_name is None:
                    raise RuntimeError(
                        f"No matching class-head output for box output {box_name} "
                        f"({h}x{w}) in {hef_path}")
                stride = self.in_h // h
                gy, gx = np.meshgrid(np.arange(h, dtype=np.float32),
                                      np.arange(w, dtype=np.float32), indexing="ij")
                cx, cy = gx + 0.5, gy + 0.5
                self._scale_pairs.append((box_name, cls_name, stride, cx, cy))
            self._scale_pairs.sort(key=lambda p: p[2])

        self.timeout_ms = timeout_ms

    def infer(self, uint8_frame):
        np.copyto(self._in_buf, uint8_frame)
        self._configured.run([self._bindings], self.timeout_ms)
        if not self._raw_dense:
            return self._bindings.output().get_buffer()
        scale_outputs = [(self._out_bufs[bn], self._out_bufs[cn], stride, cx, cy)
                          for bn, cn, stride, cx, cy in self._scale_pairs]
        return _decode_raw_dense_scales(scale_outputs, self.in_w, self.in_h)

    def close(self):
        for attr in ("_configured", "_vdevice"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "release"):
                    obj.release()
                elif hasattr(obj, "__exit__"):
                    obj.__exit__(None, None, None)
            except Exception:
                pass


def decode_detections(raw_output, orig_w, orig_h, model_w, model_h, conf_threshold):
    """raw_output: HailoBoxDetector.infer()'s per-class list.
    Returns (xyxy Nx4 float32, conf N float32, cls N int) in original-frame pixels."""
    boxes, scores, classes = [], [], []
    for class_id, class_dets in enumerate(raw_output):
        for det in class_dets:
            score = float(det[4])
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = denormalize_letterbox_box(det[:4], orig_w, orig_h, model_w, model_h)
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            classes.append(class_id)
    if not boxes:
        return (np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), int))
    return (np.array(boxes, np.float32), np.array(scores, np.float32), np.array(classes, int))


# =====================================================================
# Hailo NPU PaddleOCR pipeline (detect + recognize), vendored from
# ocr_validation_hailo_v4.py / package_ocr_usm_hailo, unchanged.
# =====================================================================
_LETTER_VALUES = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16,
    'G': 17, 'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 23,
    'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29,
    'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
    'Y': 37, 'Z': 38,
}


def compute_check_digit(container_id_10):
    if len(container_id_10) != 10:
        return None
    total = 0
    for i, ch in enumerate(container_id_10):
        val = _LETTER_VALUES[ch] if ch.isalpha() else int(ch)
        total += val * (2 ** i)
    rem = total % 11
    return 0 if rem == 10 else rem


def clean_text(text):
    text = text.strip().upper()
    text = re.sub(r"\s+", "", text)
    return text


def normalize_brightness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p_low, p_high = np.percentile(l, [1, 99])
    l_stretched = np.clip(
        (l.astype(np.float32) - p_low) * (255.0 / max(p_high - p_low, 1e-3)),
        0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_stretched, a, b]), cv2.COLOR_LAB2BGR)


def upscale_and_sharpen(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
    if APPLY_SHARPEN:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
    return img


def extract_raw_ocr_text(words):
    cleaned_words = [clean_text(w) for w in words if clean_text(w)]
    if not cleaned_words:
        return None
    return "".join(cleaned_words)


def normalize_unique_id_candidates(raw_text):
    if not raw_text:
        return []

    text = raw_text.upper()
    text = text.replace("V", "W")
    alnum = re.sub(r"[^A-Z0-9]", "", text)

    candidates = []

    for i in range(len(alnum) - 3):
        chunk = alnum[i:i + 4]
        if not all(c.isalpha() for c in chunk):
            continue

        letter_bonus = 10 if chunk[3] == "U" else (5 if chunk[3] in ("J", "Z") else 0)

        remainder = alnum[i + 4:]
        dm = re.search(r"\d{6}", remainder)
        if dm:
            digits, gap, after = dm.group(0), dm.start(), remainder[dm.end():]
        else:
            rem_fixed = remainder.replace("O", "0").replace("I", "1")
            dm = re.search(r"\d{6}", rem_fixed)
            digits, gap, after = (dm.group(0), dm.start(), rem_fixed[dm.end():]) if dm else (None, None, None)

        if digits is not None:
            digits = digits.replace("O", "0")
            score = -gap + letter_bonus
            candidate_id = chunk + digits
            expected_check = compute_check_digit(candidate_id)
            if after and after[0].isdigit() and int(after[0]) == expected_check:
                score += CHECK_DIGIT_BONUS
            candidates.append((score, candidate_id))

        before = alnum[:i]
        digit_runs = list(re.finditer(r"\d+", before))
        if not digit_runs or len(digit_runs[-1].group(0)) < 6:
            before_fixed = before.replace("O", "0").replace("I", "1")
            digit_runs = list(re.finditer(r"\d+", before_fixed))
            before = before_fixed
        if digit_runs and len(digit_runs[-1].group(0)) >= 6:
            run = digit_runs[-1]
            run_str = run.group(0)
            for k in range(len(run_str) - 5):
                digits2 = run_str[k:k + 6].replace("O", "0")
                window_end = run.start() + k + 6
                gap2 = len(before) - window_end
                score2 = -gap2 + letter_bonus
                candidate_id2 = chunk + digits2
                trailing = before[window_end:window_end + 1]
                if trailing.isdigit() and int(trailing) == compute_check_digit(candidate_id2):
                    score2 += CHECK_DIGIT_BONUS
                candidates.append((score2, candidate_id2))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def normalize_unique_id_scored(raw_text):
    candidates = normalize_unique_id_candidates(raw_text)
    if not candidates:
        return None, None
    score, candidate_id = candidates[0]
    return candidate_id, score


def pick_majority_id(track_candidates):
    if not track_candidates:
        return None, 0, 0

    scores_by_id = defaultdict(list)
    for uid, score in track_candidates:
        scores_by_id[uid].append(score)

    clusters = []
    for uid in scores_by_id:
        match = next(
            (c for c in clusters
             if any(levenshtein(uid, other) <= NEAR_DUPLICATE_MAX_DISTANCE for other in c)),
            None,
        )
        if match is None:
            clusters.append({uid})
        else:
            match.add(uid)

    def cluster_votes(cluster):
        return sum(len(scores_by_id[uid]) for uid in cluster)

    best_cluster = max(clusters, key=cluster_votes)
    winner = max(best_cluster, key=lambda uid: max(scores_by_id[uid]))

    return winner, cluster_votes(best_cluster), len(track_candidates)


def levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def reconcile_near_duplicates(entries):
    """entries: list of dicts with unique_id/track_number/votes. Keeps the
    highest-voted representative of each near-duplicate cluster."""
    ordered = sorted(entries, key=lambda e: -e["votes"])
    kept = []
    merge_log = []
    for e in ordered:
        match = next(
            (k for k in kept
             if levenshtein(e["unique_id"], k["unique_id"]) <= NEAR_DUPLICATE_MAX_DISTANCE),
            None,
        )
        if match is None:
            kept.append(e)
        else:
            merge_log.append(
                f"[MERGE] {e['unique_id']} (track {e['track_number']}, {e['votes']} votes) "
                f"looks like the same container as {match['unique_id']} "
                f"(track {match['track_number']}, {match['votes']} votes) -- keeping the latter"
            )
    return kept, merge_log


def load_dict_character(dict_path):
    with open(dict_path, "rb") as f:
        lines = f.readlines()
    chars = [line.decode("utf-8").strip("\n").strip("\r\n") for line in lines]
    return ["blank"] + chars + [" "]


def ocr_eval_postprocess(infer_results, dict_character):
    ignored_tokens = [0]

    if isinstance(infer_results, list):
        infer_results = np.array(infer_results)

    if infer_results.ndim == 2:
        infer_results = np.expand_dims(infer_results, axis=0)

    text_prob = infer_results.max(axis=2)
    text_index = infer_results.argmax(axis=2)

    batch_size = len(text_index)
    results = []

    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        char_list = [dict_character[text_id] for text_id in text_index[batch_idx][selection]]

        if text_prob is not None:
            conf_list = text_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)

        if len(conf_list) == 0:
            conf_list = [0]

        text = "".join(char_list)
        results.append((text, np.mean(conf_list).tolist()))

    return results


class DBPostProcess:
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000,
                 unclip_ratio=2.0, score_mode="fast"):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _, contours, _ = outs
        else:
            contours, _ = outs

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        h, w = bitmap.shape[:2]
        contour = np.reshape(contour.copy(), (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, heatmap, dest_height, dest_width):
        segmentation = heatmap > self.thresh
        return self.boxes_from_bitmap(heatmap, segmentation, dest_width, dest_height)


def resize_heatmap_to_original(heatmap, original_size, model_w, model_h):
    orig_h, orig_w = original_size
    scale = min(model_w / orig_w, model_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2

    cropped_heatmap = heatmap[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    return cv2.resize(cropped_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


def warp_to_rectangle(image, poly):
    poly = poly.astype(np.float32)
    w = int(np.linalg.norm(poly[0] - poly[1]))
    h = int(np.linalg.norm(poly[0] - poly[3]))
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(poly, dst_pts)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR)


def get_cropped_text_images(heatmap, orig_img, model_height, model_width):
    postprocess = DBPostProcess(
        thresh=DET_BIN_THRESH,
        box_thresh=DET_BOX_THRESH,
        max_candidates=1000,
        unclip_ratio=DET_UNCLIP_RATIO,
    )

    heatmap_resized = resize_heatmap_to_original(
        heatmap, original_size=orig_img.shape[:2],
        model_w=model_width, model_h=model_height
    )

    boxes, _scores = postprocess(heatmap_resized, *orig_img.shape[:2])

    cropped_images = []
    for box in boxes:
        try:
            box = np.array(box).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box)
            if w <= 0 or h <= 0:
                continue

            cropped = orig_img[y:y + h, x:x + w].copy()
            if cropped.size == 0:
                continue

            box = box.copy()
            box[:, 0] -= x
            box[:, 1] -= y

            mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [box], 255)
            cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

            cropped_images.append((warp_to_rectangle(cropped, box), (x, y)))
        except Exception as e:
            print(f"[WARN] Skipping bad text-region box: {e}")

    return cropped_images


def resize_with_padding(image, target_height, target_width, pad_value=128):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                              borderType=cv2.BORDER_CONSTANT, value=[pad_value] * 3)


class HailoOcrModel:
    """Synchronous single-image Hailo NPU inference wrapper (HailoRT 5.x
    InferModel API), verbatim from ocr_validation_hailo_v4.py."""

    def __init__(self, hef_path, timeout_ms=5000):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        self._vdevice = VDevice(params)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        try:
            self._infer_model.input().set_format_type(FormatType.UINT8)
        except Exception:
            pass
        try:
            self._infer_model.output().set_format_type(FormatType.FLOAT32)
        except Exception:
            pass

        self._configured = self._infer_model.configure()
        self._bindings = self._configured.create_bindings()

        ishape = self._infer_model.input().shape
        self.in_h, self.in_w = int(ishape[0]), int(ishape[1])
        self._in_buf = np.empty(ishape, dtype=np.uint8)
        self._bindings.input().set_buffer(self._in_buf)

        self._out_buf = np.empty(self._infer_model.output().shape, dtype=np.float32)
        self._bindings.output().set_buffer(self._out_buf)
        self.timeout_ms = timeout_ms

    def infer(self, uint8_frame):
        np.copyto(self._in_buf, uint8_frame)
        self._configured.run([self._bindings], self.timeout_ms)
        return self._bindings.output().get_buffer()

    def close(self):
        for attr in ("_configured", "_vdevice"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "release"):
                    obj.release()
                elif hasattr(obj, "__exit__"):
                    obj.__exit__(None, None, None)
            except Exception:
                pass


def hailo_ocr_words(bgr_image, det_model, rec_model, dict_character):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    det_input = default_preprocess(rgb, det_model.in_w, det_model.in_h)
    det_out = det_model.infer(det_input)
    heatmap = det_out[:, :, 0]

    regions = get_cropped_text_images(heatmap, rgb, det_model.in_h, det_model.in_w)
    regions.sort(key=lambda r: (r[1][1], r[1][0]))

    words = []
    for crop, _pos in regions:
        if crop.size == 0:
            continue
        rec_input = resize_with_padding(crop, rec_model.in_h, rec_model.in_w)
        rec_out = rec_model.infer(rec_input)
        text, _conf = ocr_eval_postprocess(rec_out, dict_character)[0]
        text = text.strip()
        if text:
            words.append(text)

    return words


# =====================================================================
# Drawing
# =====================================================================
def draw_label(frame, x1, y1, text, color, scale=0.55, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    ty = max(0, y1 - 6)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(frame, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def draw_stats_panel(frame, total_persons, no_helmet_count, containers_confirmed, fps):
    lines = [
        f"People: {total_persons}   No-Helmet: {no_helmet_count}   Containers ID'd: {containers_confirmed}",
        f"FPS~{fps:.1f}",
    ]
    h, w = frame.shape[:2]
    panel_h = 22 * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def write_live_frame(frame):
    h, w = frame.shape[:2]
    vis = frame
    if w > LIVE_FRAME_MAX_WIDTH:
        scale = LIVE_FRAME_MAX_WIDTH / w
        vis = cv2.resize(frame, (LIVE_FRAME_MAX_WIDTH, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, LIVE_FRAME_QUALITY])
    if not ok:
        return
    tmp_path = LIVE_FRAME_PATH + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp_path, LIVE_FRAME_PATH)


# =====================================================================
# Per-track state
# =====================================================================
class ContainerTrack:
    def __init__(self):
        self.candidates = []
        self.finalized_id = None
        self.finalized_votes = 0


# =====================================================================
# Main
# =====================================================================
def main():
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(LIVE_FRAME_PATH):
        os.remove(LIVE_FRAME_PATH)

    for path in (CONTAINER_DET_HEF_PATH, OCR_DET_HEF_PATH, OCR_REC_HEF_PATH, OCR_REC_DICT_PATH):
        if not os.path.exists(path):
            raise RuntimeError(f"Could not find required file: {path}")

    print("Loading Hailo models (NPU)...")
    dict_character = load_dict_character(OCR_REC_DICT_PATH)
    container_detector = HailoBoxDetector(CONTAINER_DET_HEF_PATH)

    # Optional: PERSON_HELMET_HEF_PATH may not exist yet (e.g. still compiling).
    # Container-ID detection/OCR runs fully either way; person/helmet detection
    # and the no-helmet count just stay empty until the hef shows up here --
    # no code change needed later, just drop the file in and restart.
    person_helmet_detector = None
    if os.path.exists(PERSON_HELMET_HEF_PATH):
        person_helmet_detector = HailoBoxDetector(PERSON_HELMET_HEF_PATH)
    else:
        print(f"[WARN] {PERSON_HELMET_HEF_PATH} not found -- running container-ID detection/OCR "
              f"only, no person/helmet detection this session.")

    ocr_det = HailoOcrModel(OCR_DET_HEF_PATH)
    ocr_rec = HailoOcrModel(OCR_REC_HEF_PATH)
    reid = ReIDEngine(REID_HEF_PATH)
    if not reid.ok:
        print("[WARN] Hailo ReID unavailable -- every re-entry counts as a new person.")
    print(f"Container detector input     : {container_detector.in_w}x{container_detector.in_h} "
          f"| classes: {CONTAINER_CLASS_NAMES}")
    if person_helmet_detector is not None:
        print(f"Person/helmet detector input : {person_helmet_detector.in_w}x{person_helmet_detector.in_h} "
              f"| classes: {PERSON_HELMET_CLASS_NAMES}")
    print(f"OCR det/rec                  : {ocr_det.in_w}x{ocr_det.in_h} / {ocr_rec.in_w}x{ocr_rec.in_h}")

    person_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)
    ocr_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)
    global_tracker = GlobalIdentityTracker(
        reid, cosine_threshold=REID_COSINE_THRESHOLD, max_samples=REID_MAX_SAMPLES,
        update_interval=REID_UPDATE_INTERVAL, track_ttl=REID_TRACK_TTL,
        new_id_confirm_attempts=REID_NEW_ID_CONFIRM_ATTEMPTS,
        min_crop_sharpness=MIN_CROP_SHARPNESS,
        adaptive_sharpness=REID_ADAPTIVE_SHARPNESS,
        sharpness_percentile=REID_SHARPNESS_PERCENTILE,
        sharpness_history_size=REID_SHARPNESS_HISTORY_SIZE,
        sharpness_warmup_samples=REID_SHARPNESS_WARMUP_SAMPLES,
        blur_gate_timeout_frames=REID_BLUR_GATE_TIMEOUT_FRAMES,
        merge_check_interval=REID_GALLERY_MERGE_CHECK_INTERVAL,
        dump_path=REID_DUMP_PATH,
    )

    person_stats = defaultdict(lambda: {"with_helmet": 0, "without_helmet": 0})  # global id -> stats
    container_tracks = defaultdict(ContainerTrack)  # local ocr track id -> state
    container_csv_rows = []  # (unique_id, track_number)
    accepted_container_entries = []  # for near-duplicate reconciliation at the end
    dedup_suppressed_total = 0  # count of head/body duplicate "person" boxes dropped (see filter_head_scale_person_boxes)
    body_ref_used_total = 0      # ReID crops that used a real paired body-box detection
    body_ref_fallback_total = 0  # ReID crops that fell back to the padded head/face crop (no paired body box found this frame)

    print("[INFO] Starting camera stream (rpicam-vid)...")
    proc = start_camera(CAM_W, CAM_H, CAM_FPS)
    print(f"[INFO] Camera started ({CAM_W}x{CAM_H}). Recording to {RECORDING_PATH}. "
          f"Running live inference... (Ctrl+C to stop)\n")

    writer = None
    frame_idx = 0
    t_start = time.time()
    last_log = 0.0
    fps_t0 = time.time()
    fps_n = 0
    fps_est = 0.0

    try:
        for frame in mjpeg_frames(proc):
            frame_idx += 1
            frame_h, frame_w = frame.shape[:2]

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(RECORDING_PATH, fourcc, float(CAM_FPS), (frame_w, frame_h))

            frame_rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------- detection (Hailo NPU, TWO separate models) ---------- #
            # Both detectors are compiled at 640x640 (same imgsz used for both
            # exports), so the letterboxed input is byte-identical for either
            # -- computing it once and reusing it skips a second full-frame
            # resize (the most expensive per-frame CPU op after JPEG decode)
            # every frame. Falls back to computing separately if a future
            # recompile ever changes one model's input size.
            container_input = default_preprocess(
                frame_rgb_full, container_detector.in_w, container_detector.in_h
            )
            container_raw = container_detector.infer(container_input)
            ocr_xyxy, ocr_conf, ocr_cls_all = decode_detections(
                container_raw, frame_w, frame_h,
                container_detector.in_w, container_detector.in_h, DET_CONF_THRESHOLD
            )
            ocr_dets = [(*box, float(conf)) for box, conf in zip(ocr_xyxy, ocr_conf)]
            ocr_cls = [int(c) for c in ocr_cls_all]

            person_boxes, helmet_boxes = [], []
            if person_helmet_detector is not None:
                if (person_helmet_detector.in_w, person_helmet_detector.in_h) == \
                        (container_detector.in_w, container_detector.in_h):
                    person_helmet_input = container_input
                else:
                    person_helmet_input = default_preprocess(
                        frame_rgb_full, person_helmet_detector.in_w, person_helmet_detector.in_h
                    )
                person_helmet_raw = person_helmet_detector.infer(person_helmet_input)
                ph_xyxy, ph_conf, ph_cls = decode_detections(
                    person_helmet_raw, frame_w, frame_h,
                    person_helmet_detector.in_w, person_helmet_detector.in_h, DET_CONF_THRESHOLD
                )
                for box, cls in zip(ph_xyxy, ph_cls):
                    cls = int(cls)
                    if cls == PERSON_IDX:
                        person_boxes.append(tuple(box))
                    elif cls == HELMET_IDX:
                        helmet_boxes.append(tuple(box))

            raw_person_count = len(person_boxes)
            person_boxes, person_body_refs = filter_head_scale_person_boxes(person_boxes, frame_h)
            dedup_suppressed_total += raw_person_count - len(person_boxes)

            has_helmet_flags = match_helmets_to_persons(person_boxes, helmet_boxes)

            # ---------------- person tracking + ReID + no-helmet voting ---------------- #
            p_dets = np.array([[*b, 1.0] for b in person_boxes], dtype=np.float32) \
                if person_boxes else np.empty((0, 5), dtype=np.float32)
            p_tracks = person_tracker.update(p_dets)

            p_det_xyxy = np.array(person_boxes, dtype=np.float32) if person_boxes else np.empty((0, 4), np.float32)
            active_gids = set()

            for trk in p_tracks:
                tx1, ty1, tx2, ty2, local_tid = trk
                local_tid = int(local_tid)
                trk_box = np.array([[tx1, ty1, tx2, ty2]])
                ious = iou_batch(trk_box, p_det_xyxy)[0] if len(p_det_xyxy) else np.array([])
                j = int(np.argmax(ious)) if len(ious) else -1
                has_helmet = has_helmet_flags[j] if (j >= 0 and ious[j] > 0.1) else False

                # ReID wants a whole-body crop, not the head/face box we
                # track/count with -- use the real paired body detection when
                # the hef fired one this frame (see filter_head_scale_person_boxes),
                # falling back to a padded crop below the head only when it
                # didn't fire.
                body_ref = person_body_refs[j] if (j >= 0 and ious[j] > 0.1) else None
                if body_ref is not None:
                    crop_rgb = crop_person_px(frame_rgb_full, *body_ref, min_crop=REID_MIN_CROP)
                    body_ref_used_total += 1
                else:
                    crop_rgb = crop_person_for_reid_px(frame_rgb_full, tx1, ty1, tx2, ty2, min_crop=REID_MIN_CROP)
                    body_ref_fallback_total += 1
                gid = global_tracker.resolve(local_tid, crop_rgb, frame_idx, active_gids)
                if gid is None:
                    # Still confirming identity (see REID_NEW_ID_CONFIRM_ATTEMPTS) --
                    # don't count this track yet, just show it's being tracked.
                    cv2.rectangle(frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), COLOR_PERSON_PENDING, 1)
                    draw_label(frame, int(tx1), int(ty1), "ID?", COLOR_PERSON_PENDING)
                    continue
                active_gids.add(gid)

                stats = person_stats[gid]
                if has_helmet:
                    stats["with_helmet"] += 1
                else:
                    stats["without_helmet"] += 1

                color = COLOR_PERSON_OK if has_helmet else COLOR_PERSON_NO_HELMET
                cv2.rectangle(frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), color, 2)
                draw_label(frame, int(tx1), int(ty1), f"G#{gid} {'OK' if has_helmet else 'NO HELMET'}", color)

            global_tracker.prune(frame_idx)
            global_tracker.check_gallery_consistency(frame_idx)
            for old_gid, new_gid in global_tracker.pop_merge_log():
                if old_gid in person_stats:
                    old_stats = person_stats.pop(old_gid)
                    new_stats = person_stats[new_gid]
                    new_stats["with_helmet"] += old_stats["with_helmet"]
                    new_stats["without_helmet"] += old_stats["without_helmet"]

            for hbox in helmet_boxes:
                x1, y1, x2, y2 = map(int, hbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HELMET, 1)

            # ---------------- OCR-class tracking + live ID extraction ---------------- #
            o_dets = np.array(ocr_dets, dtype=np.float32) if ocr_dets else np.empty((0, 5), dtype=np.float32)
            o_tracks = ocr_tracker.update(o_dets)
            o_det_xyxy = np.array([d[:4] for d in ocr_dets], dtype=np.float32) if ocr_dets else np.empty((0, 4), np.float32)

            for trk in o_tracks:
                tx1, ty1, tx2, ty2, local_tid = trk
                local_tid = int(local_tid)
                trk_box = np.array([[tx1, ty1, tx2, ty2]])
                ious = iou_batch(trk_box, o_det_xyxy)[0] if len(o_det_xyxy) else np.array([])
                j = int(np.argmax(ious)) if len(ious) else -1
                cls = ocr_cls[j] if (j >= 0 and ious[j] > 0.1) else None
                is_sharp = cls == SHARP_IDX
                color = COLOR_SHARP if is_sharp else COLOR_BLURRY
                label = f"C#{local_tid} {'sharp' if is_sharp else 'blurry'}"

                ctrack = container_tracks[local_tid]

                # Only sharp crops feed OCR/voting -- validated behavior on this
                # camera's native training-domain resolution (see docstring).
                if is_sharp and ctrack.finalized_id is None:
                    x1i = max(0, int(tx1) - CROP_PADDING)
                    y1i = max(0, int(ty1) - CROP_PADDING)
                    x2i = min(frame_w, int(tx2) + CROP_PADDING)
                    y2i = min(frame_h, int(ty2) + CROP_PADDING)
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size and (x2i - x1i) >= MIN_CROP_W and (y2i - y1i) >= MIN_CROP_H:
                        ocr_input = upscale_and_sharpen(normalize_brightness(crop))
                        words = hailo_ocr_words(ocr_input, ocr_det, ocr_rec, dict_character)
                        raw_text = extract_raw_ocr_text(words)
                        uid, score = normalize_unique_id_scored(raw_text)
                        if uid:
                            ctrack.candidates.append((uid, score))
                            winner, votes, _total = pick_majority_id(ctrack.candidates)
                            winner_validated = winner is not None and any(
                                u == winner and s >= CHECK_DIGIT_BONUS for u, s in ctrack.candidates
                            )
                            required_votes = MIN_CONTAINER_VOTES_VALIDATED if winner_validated else MIN_CONTAINER_VOTES
                            if winner is not None and votes >= required_votes:
                                ctrack.finalized_id = winner
                                ctrack.finalized_votes = votes
                                container_csv_rows.append((winner, local_tid))
                                best_score = max(s for u, s in ctrack.candidates if u == winner)
                                accepted_container_entries.append(
                                    {"unique_id": winner, "track_number": local_tid,
                                     "votes": votes, "best_score": best_score}
                                )
                                print(f"[frame {frame_idx}] CONFIRMED container id={winner} "
                                      f"(track C#{local_tid}, {votes} corroborating crop(s))")

                cv2.rectangle(frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), color, 2)
                draw_label(frame, int(tx1), int(ty1), label, color)

                if ctrack.finalized_id:
                    box_h = 30
                    if int(ty2) + 4 + box_h <= frame_h:
                        top = int(ty2) + 4
                    else:
                        top = max(0, int(ty1) - 4 - box_h)
                    bottom = top + box_h
                    cv2.rectangle(frame, (int(tx1), top), (int(tx1) + 190, bottom), COLOR_ID_CONFIRMED, -1)
                    cv2.putText(frame, ctrack.finalized_id, (int(tx1) + 4, bottom - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # ---------------- overlay + write ---------------- #
            fps_n += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps_est = fps_n / dt
                fps_n = 0
                fps_t0 = time.time()

            total_persons = len(person_stats)
            no_helmet_count = sum(
                1 for s in person_stats.values()
                if (s["with_helmet"] + s["without_helmet"]) >= MIN_OBSERVATIONS
                and s["without_helmet"] > s["with_helmet"]
            )
            containers_confirmed = sum(1 for c in container_tracks.values() if c.finalized_id is not None)
            draw_stats_panel(frame, total_persons, no_helmet_count, containers_confirmed, fps_est)

            writer.write(frame)
            if frame_idx % LIVE_FRAME_EVERY_N_FRAMES == 0:
                write_live_frame(frame)

            now = time.time()
            if now - last_log >= LOG_EVERY_SEC:
                last_log = now
                print(f"[{datetime.now().strftime('%H:%M:%S')}] frame={frame_idx} | FPS~{fps_est:.1f} | "
                      f"people={total_persons} | no_helmet={no_helmet_count} | "
                      f"containers={containers_confirmed} | dedup_suppressed={dedup_suppressed_total} | "
                      f"reentries_matched={global_tracker.total_rematches} | "
                      f"new_id_mints={global_tracker.total_new_mints} | "
                      f"blurry_skipped={global_tracker.total_blurry_skipped} | "
                      f"body_ref_used={body_ref_used_total} | "
                      f"body_ref_fallback={body_ref_fallback_total} | "
                      f"gallery_size={len(global_tracker.gallery)} | "
                      f"gallery_merges={global_tracker.total_merges} | "
                      f"sharpness_gate={global_tracker._effective_sharpness_threshold():.1f}"
                      f"(n={len(global_tracker.sharpness_history)})")
                global_tracker.save_dump()  # cheap, rewrites the file so a mid-session stop doesn't lose it

    except KeyboardInterrupt:
        print("\n[INFO] Stopping (Ctrl+C)...")

    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        if writer is not None:
            writer.release()

        container_detector.close()
        if person_helmet_detector is not None:
            person_helmet_detector.close()
        ocr_det.close()
        ocr_rec.close()
        reid.close()

        # ---------------- container ID CSV (with near-duplicate reconciliation) ---------------- #
        kept_entries, merge_log = reconcile_near_duplicates(accepted_container_entries)
        if merge_log:
            print("\n" + "\n".join(merge_log))
        kept_ids = {e["unique_id"] for e in kept_entries}
        seen_ids = set()
        dedup_rows = []
        for uid, tid in container_csv_rows:
            if uid not in kept_ids or uid in seen_ids:
                continue
            seen_ids.add(uid)
            dedup_rows.append((uid, tid))

        with open(CONTAINER_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["unique_id", "track_number"])
            for row in dedup_rows:
                writer_csv.writerow(row)
            writer_csv.writerow(["TOTAL COUNT", len(dedup_rows)])

        # ---------------- no-helmet report CSV ---------------- #
        no_helmet_rows = [
            (gid, s["without_helmet"], s["with_helmet"])
            for gid, s in person_stats.items()
            if (s["with_helmet"] + s["without_helmet"]) >= MIN_OBSERVATIONS
            and s["without_helmet"] > s["with_helmet"]
        ]
        with open(NO_HELMET_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["global_person_id", "frames_no_helmet", "frames_with_helmet"])
            for row in no_helmet_rows:
                writer_csv.writerow(row)
            writer_csv.writerow(["TOTAL UNIQUE NO-HELMET", len(no_helmet_rows), ""])

        elapsed = time.time() - t_start
        print(f"\n{'=' * 50}")
        print("Session complete")
        print(f"{'=' * 50}")
        print(f"Frames processed         : {frame_idx}")
        print(f"Elapsed time             : {elapsed:.1f}s")
        print(f"Unique people tracked    : {len(person_stats)}")
        print(f"Unique NOT wearing helmet: {len(no_helmet_rows)}")
        print(f"Unique container IDs     : {len(dedup_rows)}")
        print(f"Head/body dupes suppressed: {dedup_suppressed_total}")
        print(f"Re-entries recognized    : {global_tracker.total_rematches}")
        print(f"New identities minted    : {global_tracker.total_new_mints}")
        print(f"Blurry crops skipped     : {global_tracker.total_blurry_skipped}")
        print(f"ReID crops from body box : {body_ref_used_total}")
        print(f"ReID crops from face fallback: {body_ref_fallback_total}")
        print(f"Final gallery size       : {len(global_tracker.gallery)} people with saved embeddings")
        print(f"Duplicate identities merged: {global_tracker.total_merges}")
        global_tracker.save_dump()
        if REID_DUMP_PATH:
            print(f"ReID embedding dump      : {REID_DUMP_PATH} ({len(global_tracker.dump_rows)} rows, "
                  f"for offline threshold calibration)")
        print(f"Recording                : {RECORDING_PATH}")
        print(f"Container ID CSV         : {CONTAINER_CSV_PATH}")
        print(f"No-helmet report CSV     : {NO_HELMET_CSV_PATH}")
        print("SESSION_DONE")  # marker line for dashboard.py's wait_for_marker


if __name__ == "__main__":
    main()
