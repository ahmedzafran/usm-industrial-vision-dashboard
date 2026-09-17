"""
Hailo-accelerated person re-identification, ported from
person_det/PersonDetection/PersonDetection/basic_pipelines/detection.py so it
can run standalone inside capture_and_detect.py's own camera loop instead of
inside a GStreamer/hailo_apps pipeline callback.

Same model, same idea: the person/helmet detector's own tracker (SORT, see
capture_and_detect.py) only tracks continuously-visible people -- someone who
walks out of frame and back in gets a brand-new local track id, which would
inflate a naive "count every track" total. GlobalIdentityTracker fixes that by
mapping each local track id -> a stable "global id": whenever a local track id
has never been seen before, it embeds that person's crop with the Hailo
person-ReID network (repvgg_a0_person_reid_512, ~2ms/inference on the H10) and
matches it (cosine similarity) against a gallery of everyone seen so far in
this session. A good match reuses that person's existing global id (a
re-entry) instead of minting a new one -- which is exactly the behavior the
"no helmet" counter needs: someone already flagged and re-detected later
must not be counted twice.
"""

import pickle

import numpy as np
import cv2

try:
    import hailo_platform as hp
except ImportError:
    hp = None  # allows the rest of the module to import on a non-Hailo dev machine

REID_HEF_DEFAULT = "/usr/local/hailo/resources/models/hailo10h/repvgg_a0_person_reid_512.hef"
# Fallback shape used only if a loaded model's actual shape can't be read
# (shouldn't happen in practice) -- ReIDEngine otherwise auto-detects both
# input resolution and embedding dimension straight from the hef at load
# time (see __init__), so swapping REID_HEF_DEFAULT to a different
# person-reid model (e.g. a larger-embedding variant) doesn't need any other
# code change here.
REID_INPUT_W, REID_INPUT_H = 128, 256  # network input is 256x128x3 (HxWxC)


class ReIDEngine:
    """Wraps a Hailo person-ReID network. Degrades to a no-op if it fails to load
    (e.g. no Hailo device, or the hef isn't installed) -- callers should treat
    embed() returning None as "ReID unavailable" and fall back to tracker-only
    counting rather than crashing.

    Input resolution and output embedding dimension are read from the hef
    itself at load time (self.input_w/self.input_h/self.embedding_dim),
    not hardcoded -- this is a bigger/differently-shaped model away from
    repvgg_a0_person_reid_512 (e.g. repvgg_a0_person_reid_2048, or a
    different architecture entirely) being a drop-in hef_path swap, with no
    other code changes required."""

    def __init__(self, hef_path=REID_HEF_DEFAULT):
        self.ok = False
        self.input_w, self.input_h = REID_INPUT_W, REID_INPUT_H
        self.embedding_dim = None
        if hp is None:
            print("[ReID] hailo_platform not importable -- falling back to tracker-only counting.")
            return
        try:
            # Must match the group_id every other Hailo consumer in this
            # process uses ("SHARED" -- see HailoBoxDetector/HailoOcrModel in
            # combined_live_pi.py). combined_dashboard_pi runs container
            # detection, person/helmet detection, OCR, and ReID all
            # concurrently in one process against one physical Hailo-10H
            # device -- a mismatched group_id here means this VDevice
            # competes for the device instead of sharing it via round-robin,
            # and fails outright with HAILO_OUT_OF_PHYSICAL_DEVICES once
            # another group already holds it (confirmed live on this Pi).
            params = hp.VDevice.create_params()
            params.group_id = "SHARED"
            params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
            self.vdevice = hp.VDevice(params)
            self.model = self.vdevice.create_infer_model(hef_path)
            self.model.set_batch_size(1)
            self.model.input().set_format_type(hp.FormatType.UINT8)
            self.model.output().set_format_type(hp.FormatType.FLOAT32)
            self.cim = self.model.configure()
            self.bindings = self.cim.create_bindings()

            in_shape = tuple(int(d) for d in self.model.input().shape)   # (H, W, C)
            self.input_h, self.input_w = in_shape[0], in_shape[1]
            out_shape = tuple(int(d) for d in self.model.output().shape)
            self.embedding_dim = out_shape[-1]
            self.out_buf = np.zeros(out_shape, dtype=np.float32)

            self.ok = True
            print(f"[ReID] Hailo person-ReID model loaded from {hef_path} -- "
                  f"input {self.input_w}x{self.input_h}, embedding dim {self.embedding_dim}.")
            self.self_test()
        except Exception as e:
            print(f"[ReID] init failed -> falling back to tracker-only counting: {e}")

    def self_test(self):
        """Sanity-checks the loaded model on synthetic input, once at
        startup, so a broken model shows up immediately in the console
        instead of as mysterious duplicate-person counts later:
          1. Same image embedded twice should come back near-identical
             (cosine ~1.0) -- if not, inference isn't deterministic (buffer
             reuse bug, race condition, etc.), which would explain matches
             failing unpredictably even for the exact same crop. This check
             is trustworthy on synthetic input; keep it strict.
          2. Two different random-noise images are ALSO checked for
             similarity, but this one is informational only, not a hard
             failure signal -- confirmed on repvgg_a0_person_reid_512 on
             this exact device (2026-08-19) that pure i.i.d. random noise
             (out-of-distribution for a CNN trained on real photos, and
             INT8-quantized models are especially prone to saturating
             toward a near-constant output on structureless noise) scores
             ~0.98 here even though the SAME model discriminates real
             person crops from an actual test clip just fine (0.54-0.79
             pairwise for the same real person across different frames,
             clearly varying, not collapsed). Don't treat a high diff_sim
             here as proof the model is broken -- only real-crop evidence
             (e.g. from REID_DUMP_PATH) settles that question."""
        if not self.ok:
            return
        rng = np.random.default_rng(0)
        img_a = rng.integers(0, 255, size=(self.input_h, self.input_w, 3), dtype=np.uint8)
        img_b = rng.integers(0, 255, size=(self.input_h, self.input_w, 3), dtype=np.uint8)
        e_a1, e_a2, e_b = self.embed(img_a), self.embed(img_a), self.embed(img_b)
        if e_a1 is None or e_a2 is None or e_b is None:
            print("[ReID] self-test FAILED: embed() returned None on a synthetic input -- "
                  "ReID is effectively broken even though the model loaded.")
            return
        same_sim = float(np.dot(e_a1, e_a2))
        diff_sim = float(np.dot(e_a1, e_b))
        print(f"[ReID] self-test: same-image similarity={same_sim:.3f} (want ~1.0), "
              f"different-noise-image similarity={diff_sim:.3f} (informational -- see docstring, "
              f"quantized models can score this high on synthetic noise and still work fine on real crops)")
        if same_sim < 0.98:
            print("[ReID] WARNING: same-image similarity is low -- inference looks "
                  "non-deterministic. Re-id matching cannot be trusted until this is fixed.")

    def embed(self, crop_rgb):
        """crop_rgb: HxWx3 uint8, RGB order. Returns a unit-norm embedding
        (embedding_dim-d, whatever the loaded model actually outputs), or None."""
        if not self.ok:
            return None
        try:
            img = cv2.resize(crop_rgb, (self.input_w, self.input_h))
            self.bindings.input().set_buffer(np.ascontiguousarray(img, dtype=np.uint8))
            self.bindings.output().set_buffer(self.out_buf)
            self.cim.run([self.bindings], 10000)
            v = self.out_buf.reshape(-1).astype(np.float32).copy()
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else None
        except Exception as e:
            print(f"[ReID] embed error: {e}")
            return None

    def close(self):
        for attr in ("cim", "vdevice"):
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


def crop_person_px(frame_rgb, x1, y1, x2, y2, min_crop=24):
    """Pixel-box crop helper (our own detector already outputs pixel xyxy,
    unlike the hailo ROI bbox object the original code cropped from)."""
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < min_crop or y2 - y1 < min_crop:
        return None
    crop = frame_rgb[y1:y2, x1:x2]
    return crop if crop.size else None


def crop_sharpness(crop_rgb):
    """Variance of the Laplacian -- the standard cheap blur metric (sharp
    edges give a high-variance response, a blurred/smeared image gives a
    low one). Returns a float; higher is sharper."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


REID_TARGET_ASPECT = REID_INPUT_H / REID_INPUT_W  # 256/128 = 2.0 (H/W) -- the net expects a portrait, whole-body-shaped crop


def crop_person_for_reid_px(frame_rgb, x1, y1, x2, y2, min_crop=24,
                             target_aspect=REID_TARGET_ASPECT, side_pad_frac=0.15):
    """Like crop_person_px, but pads the box toward the ReID net's expected
    portrait aspect before cropping. Callers hand in head/face-scale
    "person" boxes -- feeding a near-square face crop straight into a
    network trained on 256x128 whole-body crops would squash it further
    than necessary. Padding sideways a bit and then downward doesn't turn a
    face into a body, but it stops the crop from being stretched even
    harder than the network already expects, and costs nothing else (this
    only changes what's fed to the embedder, never the tracked/drawn box)."""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    x1 -= side_pad_frac * w
    x2 += side_pad_frac * w
    w = x2 - x1

    desired_h = w * target_aspect
    if desired_h > h:
        y2 += (desired_h - h)

    return crop_person_px(frame_rgb, x1, y1, x2, y2, min_crop=min_crop)


class GlobalIdentityTracker:
    """Maps local (per-session) SORT track ids -> stable global person ids,
    using ReIDEngine to re-identify a track id that's never been seen before
    against everyone seen so far. Encapsulates all the bookkeeping that used
    to live directly inside detection.py's GStreamer callback.

    Usage per frame:
        gid = tracker.resolve(local_track_id, crop_rgb_or_None, frame_idx, active_gids)
        if gid is None:
            continue  # still confirming identity -- don't count this track yet
        active_gids.add(gid)
        ... at end of frame:
        tracker.prune(frame_idx)

    A brand-new local track id is NOT immediately assigned a fresh global id
    just because its first gallery match attempt misses. resolve() accumulates
    embeddings across up to new_id_confirm_attempts consecutive attempts for
    that same still-unresolved local track, re-checking the gallery with all
    of them each time (any one succeeding is enough to resolve as a
    re-entry) -- only once every attempt in that window has failed does it
    commit to "this is a new person."

    NOTE ON DESIGN HISTORY: a simpler one-shot version was tried (decide
    once, immediately, on the first crop clearing min_crop_sharpness -- no
    accumulation) since that's both the design this module was originally
    ported from (a Hailo hailo_apps GStreamer callback with no retry loop)
    and what an independent design review of this codebase recommended. It
    was tested live on this actual rig and performed WORSE, not better --
    confirmed regression, not a guess. Do not re-simplify this back to a
    one-shot decision without new evidence it would help HERE; whatever
    makes repvgg_a0_person_reid_512's embeddings well-behaved enough for a
    one-shot decision in those other setups isn't holding on this specific
    camera/lighting/crop pipeline, and accumulating multiple attempts per
    track measurably works better on it.

    check_gallery_consistency() below is a separate, complementary safety
    net -- a real-time decision (even an accumulated one) can still turn out
    wrong, and that runs periodically over the whole gallery to catch and
    merge split identities after the fact.

    The gallery (self.gallery) lives for as long as this object does --
    every embedding of every person ever seen this session stays in it for
    the whole session, including through a person walking out of frame and
    back in (prune() below only forgets the now-stale LOCAL track id, never
    touches the gallery itself, which is keyed by global id). There's
    nothing that clears it mid-session. combined_live_pi.py's process (and
    therefore this object) is spawned fresh per dashboard "Start" and killed
    on "Stop" (see dashboard.py's StageRunner), so the gallery is
    automatically wiped between sessions just by virtue of the process
    exiting -- no separate reset code needed for that.
    """

    def __init__(self, reid_engine, cosine_threshold=0.50, max_samples=8,
                 update_interval=12, track_ttl=300, new_id_confirm_attempts=60,
                 min_crop_sharpness=0.0, adaptive_sharpness=True,
                 sharpness_percentile=15, sharpness_history_size=200,
                 sharpness_warmup_samples=30, blur_gate_timeout_frames=90,
                 merge_threshold=None, merge_check_interval=30,
                 dump_path=None):
        self.reid = reid_engine
        self.cosine_threshold = cosine_threshold
        self.max_samples = max_samples
        self.update_interval = update_interval
        self.track_ttl = track_ttl
        self.new_id_confirm_attempts = new_id_confirm_attempts
        # A crop below the EFFECTIVE sharpness threshold is skipped entirely
        # -- not embedded, not spent as a confirm attempt, not added to the
        # gallery. A blurry crop doesn't just fail to match; if it's ever
        # accepted as evidence it produces a noisy/unreliable embedding that
        # then contaminates whatever it touches -- a bad match decision now,
        # or a gallery entry that's harder for THIS SAME real person's
        # future clean crops to match against later. 0.0 (default) disables
        # the gate entirely.
        #
        # min_crop_sharpness is a hard FLOOR, not the actual threshold used
        # day to day -- real crop sharpness swings wildly with scene/crop
        # framing/lighting (measured 7 to 830 on the SAME test rig depending
        # on what fell inside the crop), so any one fixed number is wrong
        # somewhere. When adaptive_sharpness is on (default), the real
        # threshold is instead the sharpness_percentile-th percentile of
        # every sharpness score actually observed this session (a rolling
        # window of the last sharpness_history_size readings) -- lenient by
        # default (15th percentile: reject only the blurriest ~15% of what's
        # typical here), and it automatically follows whatever this
        # session's real lighting/crop conditions turn out to be instead of
        # guessing them in advance. min_crop_sharpness still applies as a
        # floor underneath that (via max()) so a scene where EVERYTHING is
        # degenerate (camera fault, solid color, total wash-out) still gets
        # filtered rather than the adaptive threshold chasing it down to
        # near-zero. Before sharpness_warmup_samples readings exist there's
        # no real distribution to calibrate against yet, so the gate stays
        # fully open (accepts everything) rather than guessing.
        self.min_crop_sharpness = min_crop_sharpness
        self.adaptive_sharpness = adaptive_sharpness
        self.sharpness_percentile = sharpness_percentile
        self.sharpness_warmup_samples = sharpness_warmup_samples
        self.sharpness_history = []  # rolling window, capped at sharpness_history_size, oldest dropped first
        self.sharpness_history_size = sharpness_history_size
        self.total_blurry_skipped = 0  # count of crops rejected by the sharpness gate
        # Safety valve: if a still-unresolved local track has gone this many
        # frames WITHOUT EVER recording a single real attempt (every crop it
        # ever produced failed the sharpness gate), the gate stops blocking
        # it and starts accepting whatever crop is available. Without this,
        # a person under dim/low-contrast/soft lighting where no crop ever
        # clears min_crop_sharpness would be stuck showing "ID?" and never
        # get counted, for their entire time in frame, no matter how long
        # that is -- confirmed live: a whole ~28s session with one
        # continuously-visible person, zero crops ever cleared the gate,
        # and person_stats stayed empty the entire time. This bounds how
        # long that lockout can last.
        self.blur_gate_timeout_frames = blur_gate_timeout_frames
        self.track_first_seen = {}   # local track id -> frame_idx first seen still-unresolved
        self.blur_gate_bypassed = set()  # local track ids the gate has given up on -- once tripped, stays off for that track (a one-frame-only bypass would just re-arm the instant the resulting attempt got recorded, defeating the point)

        # Real-time matching (resolve()) only gets ONE shot at a decision --
        # under blur/bad angle that decision can still be wrong, splitting
        # one real person into two gallery entries. check_gallery_consistency()
        # below is a second pass over the WHOLE gallery (call periodically,
        # not per-frame) that catches that after the fact: two "different"
        # identities whose accumulated samples cross-match well are almost
        # certainly the same person, and get folded into one. Defaults to
        # cosine_threshold if not given explicitly.
        self.merge_threshold = merge_threshold if merge_threshold is not None else cosine_threshold
        self.merge_check_interval = merge_check_interval
        self._last_merge_check_frame = -merge_check_interval
        self.merge_log = []      # [(old_gid, new_gid), ...] since the last pop_merge_log() -- callers with their own per-gid bookkeeping (e.g. helmet stats) should apply the same remap
        self.total_merges = 0

        # If set, every embedding resolve() computes gets appended here as
        # (frame_idx, local_track_id, embedding_list) and can be flushed to
        # disk with save_dump() -- collected purely for offline threshold
        # calibration (comparing same-track vs. different-track cosine
        # scores from a real session) rather than guessing REID_COSINE_THRESHOLD
        # from anecdotal live results.
        self.dump_path = dump_path
        self.dump_rows = []

        self._next_global_id = 0
        self.track_to_global = {}     # local track id -> global id
        self.gallery = {}             # global id -> list[embedding] -- persists for the whole session, see class docstring
        self.gallery_updated = {}     # global id -> frame idx of last embedding
        self.track_last_seen = {}     # local track id -> frame idx last seen
        # local track id -> {"embs": [...capped at max_samples...], "attempts": total count}
        # attempts is tracked separately from embs because max_samples can be
        # (and by default is) smaller than new_id_confirm_attempts -- embs is
        # just what's kept around for matching, attempts is the real tally
        # resolve() checks against the confirm threshold.
        self.pending = {}
        self.seen_global_ids = set()  # every distinct person ever seen this session
        self.total_rematches = 0      # count of resolve() calls that reused an existing gallery id (a real re-entry recognized)
        self.total_new_mints = 0      # count of resolve() calls that minted a brand-new global id

    def _new_global_id(self):
        self._next_global_id += 1
        return self._next_global_id

    def _add_sample(self, gid, emb, frame_idx):
        if emb is None:
            return
        embs = self.gallery.setdefault(gid, [])
        embs.append(emb)
        if len(embs) > self.max_samples:
            embs.pop(0)
        self.gallery_updated[gid] = frame_idx

    def _due_update(self, gid, frame_idx):
        return frame_idx - self.gallery_updated.get(gid, -self.update_interval) >= self.update_interval

    def _record_sharpness(self, value):
        self.sharpness_history.append(value)
        if len(self.sharpness_history) > self.sharpness_history_size:
            self.sharpness_history.pop(0)

    def _effective_sharpness_threshold(self):
        """The real gate value for this moment: a lenient percentile of
        every sharpness reading actually seen recently, floored at
        min_crop_sharpness. Returns 0.0 (gate wide open) if adaptive mode
        is off, or there isn't enough history yet to calibrate against."""
        if not self.adaptive_sharpness:
            return self.min_crop_sharpness
        if len(self.sharpness_history) < self.sharpness_warmup_samples:
            return 0.0
        percentile_value = float(np.percentile(self.sharpness_history, self.sharpness_percentile))
        return max(self.min_crop_sharpness, percentile_value)

    def _cross_gid_score(self, gid_a, gid_b):
        """Similarity between two established identities' full sample sets.
        Uses the mean of the top-2 pairwise scores rather than a bare max --
        with up to max_samples^2 pairwise comparisons, a raw max is prone to
        one lucky/noisy outlier crossing the threshold purely by chance; a
        real duplicate identity should have SEVERAL strong cross-matches,
        not just one."""
        scores = []
        for ea in self.gallery.get(gid_a, []):
            for eb in self.gallery.get(gid_b, []):
                scores.append(float(np.dot(ea, eb)))
        if not scores:
            return -1.0
        scores.sort(reverse=True)
        top = scores[:2]
        return sum(top) / len(top)

    def _merge_identity(self, old_gid, new_gid):
        """Folds old_gid into new_gid everywhere: gallery samples, bookkeeping,
        and any LOCAL track currently mapped to old_gid (so the very next
        frame for that still-active track already resolves to new_gid, not
        the retired id)."""
        if old_gid == new_gid:
            return
        old_embs = self.gallery.pop(old_gid, [])
        merged = self.gallery.setdefault(new_gid, []) + old_embs
        self.gallery[new_gid] = merged[-self.max_samples:]
        self.gallery_updated[new_gid] = max(
            self.gallery_updated.get(new_gid, 0), self.gallery_updated.pop(old_gid, 0)
        )
        for local_tid, gid in list(self.track_to_global.items()):
            if gid == old_gid:
                self.track_to_global[local_tid] = new_gid
        self.seen_global_ids.discard(old_gid)
        self.seen_global_ids.add(new_gid)
        self.merge_log.append((old_gid, new_gid))
        self.total_merges += 1
        print(f"[ReID] gallery self-check: merged duplicate identity gid={old_gid} "
              f"into gid={new_gid} (their samples cross-match well enough to be the same person)")

    def check_gallery_consistency(self, frame_idx):
        """Periodically (every merge_check_interval frames -- this is O(number
        of distinct identities squared) so it's throttled, not run every
        frame) scans every pair of DIFFERENT gallery identities for
        cross-similarity above merge_threshold and merges any that qualify,
        keeping the smaller (earlier-created) id as canonical. See the class
        docstring / __init__ comment for why this matters: real-time
        matching only gets one noisy shot at a decision, this gets to look
        at the accumulated evidence for both identities at once.

        Callers that keep their own per-gid bookkeeping (helmet stats, etc.)
        must call pop_merge_log() after this and apply the same (old_gid ->
        new_gid) remaps to their own data, or a merged identity's earlier
        stats will silently vanish instead of combining."""
        if frame_idx - self._last_merge_check_frame < self.merge_check_interval:
            return
        self._last_merge_check_frame = frame_idx

        gids = sorted(self.gallery.keys())
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                a, b = gids[i], gids[j]
                if a not in self.gallery or b not in self.gallery:
                    continue  # one side already merged away earlier in this same pass
                if self._cross_gid_score(a, b) >= self.merge_threshold:
                    old, new = (b, a) if a < b else (a, b)
                    self._merge_identity(old, new)

    def pop_merge_log(self):
        """Returns and clears the list of (old_gid, new_gid) merges applied
        since the last call -- callers with their own per-gid bookkeeping
        should fold old_gid's data into new_gid's for each entry."""
        log, self.merge_log = self.merge_log, []
        return log

    def _match_gallery(self, emb, exclude_ids):
        best_id, best = None, -1.0
        for gid, embs in self.gallery.items():
            if gid in exclude_ids:
                continue
            for e in embs:
                s = float(np.dot(emb, e))
                if s > best:
                    best, best_id = s, gid
        return best_id, best

    def _dump(self, local_track_id, frame_idx, emb):
        if self.dump_path is not None and emb is not None:
            self.dump_rows.append((frame_idx, local_track_id, emb.tolist()))

    def save_dump(self):
        """Flushes collected (frame_idx, local_track_id, embedding) rows to
        dump_path for offline threshold calibration. Cheap to call
        periodically (e.g. alongside the console log) plus once more at
        session end -- overwrites the file each time with everything
        collected so far, so a mid-session crash doesn't lose the data."""
        if not self.dump_path:
            return
        try:
            with open(self.dump_path, "wb") as f:
                pickle.dump(self.dump_rows, f)
        except Exception as e:
            print(f"[ReID] dump failed: {e}")

    def resolve(self, local_track_id, crop_rgb, frame_idx, active_gids):
        """Returns the global id for this local track id this frame, or
        None if it's still in its confirmation window (see class docstring)
        -- callers must skip counting for a None result rather than treating
        it as an id.

        Skips the Hailo embed call entirely for tracks that are already
        resolved and not yet due for a gallery refresh -- ReID cost is paid
        only for genuinely new/pending tracks or periodic refreshes, not
        every track every frame.
        """
        gid = self.track_to_global.get(local_track_id)
        if gid is not None and not self._due_update(gid, frame_idx):
            self.seen_global_ids.add(gid)
            self.track_last_seen[local_track_id] = frame_idx
            return gid

        if gid is None and local_track_id not in self.track_first_seen:
            self.track_first_seen[local_track_id] = frame_idx

        usable_crop = crop_rgb is not None
        gate_active = self.adaptive_sharpness or self.min_crop_sharpness > 0.0
        if usable_crop and gate_active and local_track_id not in self.blur_gate_bypassed:
            sharpness = crop_sharpness(crop_rgb)
            self._record_sharpness(sharpness)  # record every reading regardless of pass/fail -- the adaptive threshold needs the true distribution, not just the accepted tail
            effective_threshold = self._effective_sharpness_threshold()
            if sharpness < effective_threshold:
                has_any_attempt = local_track_id in self.pending and self.pending[local_track_id]["attempts"] > 0
                waited_blind = frame_idx - self.track_first_seen.get(local_track_id, frame_idx)
                if gid is None and not has_any_attempt and waited_blind >= self.blur_gate_timeout_frames:
                    self.blur_gate_bypassed.add(local_track_id)
                    print(f"[ReID] track {local_track_id}: no crop cleared the sharpness gate "
                          f"(effective threshold={effective_threshold:.1f}) after {waited_blind} "
                          f"frames -- bypassing the gate for this track from now on so it can resolve")
                else:
                    usable_crop = False
                    self.total_blurry_skipped += 1

        emb = self.reid.embed(crop_rgb) if (self.reid.ok and usable_crop) else None
        if emb is not None:
            self._dump(local_track_id, frame_idx, emb)

        if gid is None:
            if not self.reid.ok:
                # No re-id signal will ever arrive for this track -- waiting
                # would just delay counting forever, so commit immediately
                # (matches the pre-existing "tracker-only counting" fallback).
                gid = self._new_global_id()
                self.track_to_global[local_track_id] = gid
                self.pending.pop(local_track_id, None)
                self.total_new_mints += 1
            elif emb is not None:
                pend = self.pending.get(local_track_id)
                if pend is None:
                    pend = {"embs": [], "attempts": 0}
                    self.pending[local_track_id] = pend
                pend["embs"].append(emb)
                if len(pend["embs"]) > self.max_samples:
                    pend["embs"].pop(0)
                pend["attempts"] += 1

                # Any one of this track's accumulated attempts crossing the
                # threshold is enough -- a later, cleaner crop of the same
                # still-unresolved person can succeed even if an earlier one
                # didn't.
                best_cand, best_score = None, -1.0
                for e in pend["embs"]:
                    cand, score = self._match_gallery(e, active_gids)
                    if score > best_score:
                        best_cand, best_score = cand, score

                if best_cand is not None and best_score >= self.cosine_threshold:
                    gid = best_cand
                    self.track_to_global[local_track_id] = gid
                    for e in pend["embs"]:
                        self._add_sample(gid, e, frame_idx)
                    self.pending.pop(local_track_id, None)
                    self.total_rematches += 1
                    print(f"[ReID] track {local_track_id}: matched existing gid={gid} "
                          f"score={best_score:.3f} after {pend['attempts']} attempts -- "
                          f"re-entry recognized")
                elif pend["attempts"] >= self.new_id_confirm_attempts:
                    gid = self._new_global_id()
                    self.track_to_global[local_track_id] = gid
                    for e in pend["embs"]:
                        self._add_sample(gid, e, frame_idx)
                    self.pending.pop(local_track_id, None)
                    self.total_new_mints += 1
                    print(f"[ReID] track {local_track_id}: no gallery match after "
                          f"{pend['attempts']} attempts (best gid={best_cand} "
                          f"score={best_score:.3f} < threshold={self.cosine_threshold:.2f}) "
                          f"-> minting new global id {gid}")
                else:
                    print(f"[ReID] track {local_track_id}: attempt {pend['attempts']}/"
                          f"{self.new_id_confirm_attempts}, best match gid={best_cand} "
                          f"score={best_score:.3f} < threshold={self.cosine_threshold:.2f} "
                          f"-- waiting for more evidence")
                    self.track_last_seen[local_track_id] = frame_idx
                    return None
            else:
                # reid.ok but no usable crop this frame (box too small/off-
                # frame, or too blurry per min_crop_sharpness) -- stay
                # pending without spending an attempt, retry once a valid
                # crop shows up rather than guessing now.
                self.track_last_seen[local_track_id] = frame_idx
                return None
        elif emb is not None:
            self._add_sample(gid, emb, frame_idx)

        self.track_first_seen.pop(local_track_id, None)
        self.blur_gate_bypassed.discard(local_track_id)
        self.seen_global_ids.add(gid)
        self.track_last_seen[local_track_id] = frame_idx
        return gid

    def prune(self, frame_idx):
        stale = [t for t, seen in self.track_last_seen.items() if frame_idx - seen > self.track_ttl]
        for t in stale:
            self.track_last_seen.pop(t, None)
            self.track_to_global.pop(t, None)
            self.pending.pop(t, None)
            self.track_first_seen.pop(t, None)
            self.blur_gate_bypassed.discard(t)

    def total_count(self):
        return len(self.seen_global_ids)
