#!/usr/bin/env python3
"""
Same fully Hailo-accelerated pipeline as combined_live_pi.py, but reading
from pre-recorded video files (input_videos/*.mp4 etc.) instead of the live
camera -- for testing/tuning against a fixed, repeatable dataset instead of
whatever happens to walk in front of the camera right now.

All the model loading, detection, tracking, ReID, and OCR logic is imported
directly from combined_live_pi.py rather than reimplemented here, so any fix
made there (e.g. the raw-dense-output decode workaround, or the latency
optimizations) applies to both scripts automatically.

Usage:
    mkdir -p input_videos
    # drop .mp4/.avi/.mov/.mkv files into input_videos/
    python3 combined_video_file_pi.py

Outputs, per input video, land in video_outputs/:
    <name>_annotated.mp4
    <name>_container_ids.csv
    <name>_no_helmet_report.csv
plus a combined video_outputs/summary.csv across all processed videos.
"""

import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

from combined_live_pi import (
    CONTAINER_DET_HEF_PATH, PERSON_HELMET_HEF_PATH, OCR_DET_HEF_PATH,
    OCR_REC_HEF_PATH, OCR_REC_DICT_PATH,
    BLURRY_IDX, SHARP_IDX, CONTAINER_CLASS_NAMES,
    HELMET_IDX, PERSON_IDX, PERSON_HELMET_CLASS_NAMES,
    DET_CONF_THRESHOLD,
    SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESHOLD,
    HEAD_ZONE_UPWARD_PAD, MIN_HEAD_OVERLAP, MIN_OBSERVATIONS,
    MIN_CONTAINER_VOTES, MIN_CONTAINER_VOTES_VALIDATED, CHECK_DIGIT_BONUS,
    REID_HEF_PATH, REID_COSINE_THRESHOLD, REID_MAX_SAMPLES,
    REID_UPDATE_INTERVAL, REID_TRACK_TTL, REID_MIN_CROP, REID_NEW_ID_CONFIRM_ATTEMPTS,
    MIN_CROP_SHARPNESS,
    CROP_PADDING, MIN_CROP_W, MIN_CROP_H,
    COLOR_PERSON_OK, COLOR_PERSON_NO_HELMET, COLOR_PERSON_PENDING, COLOR_HELMET,
    COLOR_BLURRY, COLOR_SHARP, COLOR_ID_CONFIRMED,
    iou_batch, Sort, match_helmets_to_persons, filter_head_scale_person_boxes,
    default_preprocess, HailoBoxDetector, decode_detections,
    normalize_brightness, upscale_and_sharpen, extract_raw_ocr_text,
    normalize_unique_id_scored, pick_majority_id, reconcile_near_duplicates,
    load_dict_character, HailoOcrModel, hailo_ocr_words,
    ReIDEngine, GlobalIdentityTracker, crop_person_px, crop_person_for_reid_px,
    draw_label, draw_stats_panel, ContainerTrack,
)

INPUT_DIR = "input_videos"
OUTPUT_DIR = "video_outputs"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
LOG_EVERY_SEC = 5.0


def process_video(video_path, container_detector, person_helmet_detector,
                   ocr_det, ocr_rec, dict_character, reid,
                   out_video_path, out_container_csv_path, out_no_helmet_csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, src_fps, (frame_w, frame_h))

    person_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)
    ocr_tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)
    global_tracker = GlobalIdentityTracker(
        reid, cosine_threshold=REID_COSINE_THRESHOLD, max_samples=REID_MAX_SAMPLES,
        update_interval=REID_UPDATE_INTERVAL, track_ttl=REID_TRACK_TTL,
        new_id_confirm_attempts=REID_NEW_ID_CONFIRM_ATTEMPTS,
        min_crop_sharpness=MIN_CROP_SHARPNESS,
    )

    person_stats = defaultdict(lambda: {"with_helmet": 0, "without_helmet": 0})
    container_tracks = defaultdict(ContainerTrack)
    container_csv_rows = []
    accepted_container_entries = []
    dedup_suppressed_total = 0  # count of head/body duplicate "person" boxes dropped (see filter_head_scale_person_boxes)

    frame_idx = 0
    t_start = time.time()
    last_log = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- detection (Hailo NPU, TWO separate models) ---------- #
        # Both detectors are compiled at 640x640, so the letterboxed input is
        # byte-identical for either -- computed once and reused (see
        # combined_live_pi.py for the full rationale).
        container_input = default_preprocess(frame_rgb, container_detector.in_w, container_detector.in_h)
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
                    frame_rgb, person_helmet_detector.in_w, person_helmet_detector.in_h
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

            body_ref = person_body_refs[j] if (j >= 0 and ious[j] > 0.1) else None
            if body_ref is not None:
                crop_rgb = crop_person_px(frame_rgb, *body_ref, min_crop=REID_MIN_CROP)
            else:
                crop_rgb = crop_person_for_reid_px(frame_rgb, tx1, ty1, tx2, ty2, min_crop=REID_MIN_CROP)
            gid = global_tracker.resolve(local_tid, crop_rgb, frame_idx, active_gids)
            if gid is None:
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
        total_persons = len(person_stats)
        no_helmet_count = sum(
            1 for s in person_stats.values()
            if (s["with_helmet"] + s["without_helmet"]) >= MIN_OBSERVATIONS
            and s["without_helmet"] > s["with_helmet"]
        )
        containers_confirmed = sum(1 for c in container_tracks.values() if c.finalized_id is not None)
        elapsed = time.time() - t_start
        fps_est = frame_idx / elapsed if elapsed > 0 else 0.0
        draw_stats_panel(frame, total_persons, no_helmet_count, containers_confirmed, fps_est)

        writer.write(frame)

        now = time.time()
        if now - last_log >= LOG_EVERY_SEC:
            last_log = now
            pct = (frame_idx / total_frames * 100) if total_frames else 0.0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {os.path.basename(video_path)} "
                  f"frame={frame_idx}/{total_frames} ({pct:.0f}%) | FPS~{fps_est:.1f} | "
                  f"people={total_persons} | no_helmet={no_helmet_count} | "
                  f"containers={containers_confirmed} | dedup_suppressed={dedup_suppressed_total}")

    cap.release()
    writer.release()

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

    with open(out_container_csv_path, "w", newline="", encoding="utf-8") as f:
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
    with open(out_no_helmet_csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["global_person_id", "frames_no_helmet", "frames_with_helmet"])
        for row in no_helmet_rows:
            writer_csv.writerow(row)
        writer_csv.writerow(["TOTAL UNIQUE NO-HELMET", len(no_helmet_rows), ""])

    return {
        "video": os.path.basename(video_path),
        "frames": frame_idx,
        "elapsed_sec": round(time.time() - t_start, 1),
        "unique_people": len(person_stats),
        "unique_no_helmet": len(no_helmet_rows),
        "unique_container_ids": len(dedup_rows),
        "dedup_suppressed": dedup_suppressed_total,
    }


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        print(f"[INFO] No video files found in {INPUT_DIR}/ "
              f"(looking for {sorted(VIDEO_EXTENSIONS)}). Nothing to do.")
        return

    for path in (CONTAINER_DET_HEF_PATH, OCR_DET_HEF_PATH, OCR_REC_HEF_PATH, OCR_REC_DICT_PATH):
        if not os.path.exists(path):
            raise RuntimeError(f"Could not find required file: {path}")

    print("Loading Hailo models (NPU)...")
    dict_character = load_dict_character(OCR_REC_DICT_PATH)
    container_detector = HailoBoxDetector(CONTAINER_DET_HEF_PATH)

    person_helmet_detector = None
    if os.path.exists(PERSON_HELMET_HEF_PATH):
        person_helmet_detector = HailoBoxDetector(PERSON_HELMET_HEF_PATH)
    else:
        print(f"[WARN] {PERSON_HELMET_HEF_PATH} not found -- running container-ID detection/OCR "
              f"only, no person/helmet detection this run.")

    ocr_det = HailoOcrModel(OCR_DET_HEF_PATH)
    ocr_rec = HailoOcrModel(OCR_REC_HEF_PATH)
    reid = ReIDEngine(REID_HEF_PATH)
    if not reid.ok:
        print("[WARN] Hailo ReID unavailable -- every re-entry counts as a new person.")

    print(f"Found {len(video_files)} video(s) in {INPUT_DIR}/: {video_files}")

    summary_rows = []
    try:
        for fname in video_files:
            video_path = os.path.join(INPUT_DIR, fname)
            stem = os.path.splitext(fname)[0]
            out_video_path = os.path.join(OUTPUT_DIR, f"{stem}_annotated.mp4")
            out_container_csv_path = os.path.join(OUTPUT_DIR, f"{stem}_container_ids.csv")
            out_no_helmet_csv_path = os.path.join(OUTPUT_DIR, f"{stem}_no_helmet_report.csv")

            print(f"\n{'=' * 50}\nProcessing {fname}\n{'=' * 50}")
            result = process_video(
                video_path, container_detector, person_helmet_detector,
                ocr_det, ocr_rec, dict_character, reid,
                out_video_path, out_container_csv_path, out_no_helmet_csv_path,
            )
            summary_rows.append(result)
            print(f"[DONE] {fname}: {result}")
    finally:
        container_detector.close()
        if person_helmet_detector is not None:
            person_helmet_detector.close()
        ocr_det.close()
        ocr_rec.close()
        reid.close()

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["video", "frames", "elapsed_sec", "unique_people",
                              "unique_no_helmet", "unique_container_ids", "dedup_suppressed"])
        for row in summary_rows:
            writer_csv.writerow([row["video"], row["frames"], row["elapsed_sec"],
                                  row["unique_people"], row["unique_no_helmet"],
                                  row["unique_container_ids"], row["dedup_suppressed"]])

    print(f"\n{'=' * 50}")
    print("All videos processed")
    print(f"{'=' * 50}")
    for row in summary_rows:
        print(f"  {row['video']}: people={row['unique_people']} "
              f"no_helmet={row['unique_no_helmet']} containers={row['unique_container_ids']} "
              f"dedup_suppressed={row['dedup_suppressed']}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
