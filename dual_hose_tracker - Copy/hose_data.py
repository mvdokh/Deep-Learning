"""
Dual hose keypoint data loading and video inference (L, R only).

Expected session layout under ``data_folder``::

    data_folder/
      <session>/
        images/
        hoseL.csv
        hoseR.csv

Training policy
---------------
- Frames with BOTH L and R  -> two Gaussian heatmaps
- Frames with NEITHER       -> empty heatmaps (null / no hose)
- Exactly one of L/R        -> discarded

Label convention (hose-relative, NOT screen x-order)
---------------------------------------------------
Reference pose: hose pointing toward the **bottom** of the screen.
``hoseL`` / ``hoseR`` are left / right edges in that hose frame.
When the hose points up, ``hoseL`` may sit to the right of ``hoseR`` on
screen — that is correct; do not swap coords.

For the 4-keypoint (L,R,T,B) pipeline see ``../quad_hose_tracker/``.
"""

from __future__ import annotations

import csv
import os
import re
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from src.DeepLearningUtils.DataStructures.Keypoints.keypoint import Keypoint, VideoKeypoints
from src.DeepLearningUtils.utils.data_conversion import convert_img_numpy
from src.DeepLearningUtils.utils.image_processing.processing import create_gaussian_mask
from src.DeepLearningUtils.utils.video.video_opencv import get_total_frames, get_video_height_width


KEYPOINT_NAMES = ("hoseL", "hoseR")
DEFAULT_CSV_NAMES = {
    "hoseL": "hoseL.csv",
    "hoseR": "hoseR.csv",
}
DEFAULT_FRAME_REGEX = r"img(\d+)"
EXCLUSIVE_PAIRS = ((0, 1),)


def _extract_frame_number(filename: str, frame_number_regex: Optional[str]) -> Union[int, str]:
    name = os.path.splitext(os.path.basename(filename))[0]
    if frame_number_regex:
        match = re.search(frame_number_regex, name)
        if match:
            name = match.group(1)
    try:
        return int(name)
    except ValueError:
        return name


def _read_keypoint_csv(
    csv_path: Path,
    delimiter: Optional[str] = None,
    has_header: bool = True,
    occlusion_markers: Tuple[str, ...] = ("nan", "NaN", "NAN", "None", ""),
) -> Dict[Union[int, str], Optional[List[int]]]:
    coords: Dict[Union[int, str], Optional[List[int]]] = {}
    if not csv_path.is_file():
        return coords

    with open(csv_path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)

        for row in reader:
            if not row or len(row) < 3:
                continue
            frame_raw = row[0].strip()
            try:
                frame_num: Union[int, str] = int(float(frame_raw))
            except ValueError:
                frame_num = frame_raw

            x_raw, y_raw = str(row[1]).strip(), str(row[2]).strip()
            if x_raw in occlusion_markers or y_raw in occlusion_markers:
                coords[frame_num] = None
                continue
            try:
                coords[frame_num] = [int(float(x_raw)), int(float(y_raw))]
            except ValueError:
                print(f"Skipping invalid coords in {csv_path}: {row}")
    return coords


def audit_hose_label_order(
    data_folder: str,
    csv_names: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, int]]:
    """Report Lx>Rx mix (often hose pointing up under bottom-ref convention)."""
    data_folder = str(data_folder)
    csv_names = csv_names or dict(DEFAULT_CSV_NAMES)
    report: Dict[str, Dict[str, int]] = {}

    for session in sorted(os.listdir(data_folder)):
        session_path = Path(data_folder) / session
        if not session_path.is_dir():
            continue
        left_csv = session_path / csv_names["hoseL"]
        right_csv = session_path / csv_names["hoseR"]
        if not left_csv.is_file() or not right_csv.is_file():
            continue

        left_coords = _read_keypoint_csv(left_csv)
        right_coords = _read_keypoint_csv(right_csv)
        common = set(left_coords) & set(right_coords)
        l_x_gt_r = 0
        n_valid = 0
        for frame in common:
            lc, rc = left_coords[frame], right_coords[frame]
            if lc is None or rc is None:
                continue
            n_valid += 1
            if lc[0] > rc[0]:
                l_x_gt_r += 1
        report[session] = {
            "labeled": n_valid,
            "l_x_gt_r": l_x_gt_r,
            "l_x_le_r": n_valid - l_x_gt_r,
        }
    return report


def _winner_take_all_channels(label: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label)
    if not np.any(label > 0):
        return out
    winner = np.argmax(label, axis=-1)
    for c in range(label.shape[-1]):
        mask = winner == c
        out[..., c][mask] = label[..., c][mask]
    return out


def load_hose_data(
    data_folder: str,
    target_resolution: Tuple[int, int] = (256, 256),
    gaussian_sigma: Tuple[int, int] = (11, 11),
    csv_names: Optional[Dict[str, str]] = None,
    frame_number_regex: Optional[str] = DEFAULT_FRAME_REGEX,
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    images_dir_name: str = "images",
    include_null_frames: bool = True,
    require_both: bool = True,
    max_null_ratio: Optional[float] = 1.0,
    null_seed: int = 4,
    return_numpy: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load L/R hose training data.

    Returns labels (N, H, W, 2) = hoseL, hoseR.
    With ``require_both=True``, frames with only one side labeled are discarded.
    """
    data_folder = str(data_folder)
    csv_names = csv_names or dict(DEFAULT_CSV_NAMES)
    keypoint_names = list(KEYPOINT_NAMES)

    session_folders = sorted(
        name
        for name in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, name))
    )
    if not session_folders:
        raise FileNotFoundError(f"No session folders found under {data_folder}")

    training_images: List[np.ndarray] = []
    training_filenames: List[str] = []
    training_labels: List[np.ndarray] = []

    n_kept_both = 0
    n_kept_null = 0
    n_discarded_partial = 0
    n_missing_image = 0

    for session in tqdm(session_folders, desc="Loading sessions", ascii=True):
        session_path = Path(data_folder) / session
        img_folder = session_path / images_dir_name
        if not img_folder.is_dir():
            print(f"Skipping {session}: no '{images_dir_name}/' folder")
            continue

        csv_paths = {k: session_path / csv_names[k] for k in keypoint_names}
        missing = [k for k, p in csv_paths.items() if not p.is_file()]
        if missing:
            print(f"Skipping {session}: missing CSVs {missing}")
            continue

        all_coords = {k: _read_keypoint_csv(csv_paths[k]) for k in keypoint_names}

        image_paths = sorted(
            p
            for p in img_folder.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )
        if not image_paths:
            print(f"Skipping {session}: no images")
            continue

        original_resolution = None
        for probe in image_paths:
            img0 = cv2.imread(str(probe))
            if img0 is not None:
                original_resolution = (img0.shape[0], img0.shape[1])
                break
        if original_resolution is None:
            print(f"Skipping {session}: could not read any images")
            continue

        print(
            f"{session}: {len(image_paths)} images, "
            f"L={len(all_coords['hoseL'])}, R={len(all_coords['hoseR'])}, "
            f"resolution={original_resolution}"
        )

        for image_path in image_paths:
            frame_num = _extract_frame_number(image_path.name, frame_number_regex)
            present = {
                k: (frame_num in all_coords[k] and all_coords[k][frame_num] is not None)
                for k in keypoint_names
            }
            n_present = sum(present.values())

            if n_present == 2:
                pass
            elif n_present == 0:
                if not include_null_frames:
                    continue
            else:
                if require_both:
                    n_discarded_partial += 1
                    continue

            image = cv2.imread(str(image_path))
            if image is None:
                n_missing_image += 1
                continue

            image_resized = cv2.resize(
                image,
                (target_resolution[1], target_resolution[0]),
                interpolation=cv2.INTER_AREA,
            )

            channel_masks = []
            for k in keypoint_names:
                if present[k]:
                    mask = create_gaussian_mask(
                        original_resolution,
                        target_resolution,
                        all_coords[k][frame_num],
                        gaussian_sigma,
                    )
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = np.zeros(target_resolution, dtype=np.uint8)
                channel_masks.append(mask)

            label = np.stack(channel_masks, axis=-1).astype(np.float32)
            label = _winner_take_all_channels(label).astype(np.uint8)

            training_images.append(image_resized)
            training_filenames.append(str(image_path))
            training_labels.append(label)

            if n_present == 2:
                n_kept_both += 1
            else:
                n_kept_null += 1

    print(
        f"Loaded {len(training_images)} frames "
        f"(both={n_kept_both}, null={n_kept_null}, "
        f"discarded_partial={n_discarded_partial}, unreadable={n_missing_image})"
    )

    if max_null_ratio is not None and n_kept_null > 0 and n_kept_both > 0:
        null_limit = int(np.floor(max_null_ratio * n_kept_both))
        if n_kept_null > null_limit:
            labeled_idx = [
                i for i, lab in enumerate(training_labels) if np.any(lab > 0)
            ]
            null_idx = [
                i for i, lab in enumerate(training_labels) if not np.any(lab > 0)
            ]
            rng = np.random.default_rng(null_seed)
            keep_null = rng.choice(null_idx, size=null_limit, replace=False).tolist()
            keep = sorted(labeled_idx + keep_null)
            training_images = [training_images[i] for i in keep]
            training_filenames = [training_filenames[i] for i in keep]
            training_labels = [training_labels[i] for i in keep]
            print(
                f"Balanced nulls: kept {null_limit}/{n_kept_null} null frames "
                f"-> {len(training_images)} total"
            )

    print(f"Keypoint channels: {keypoint_names}")

    if not training_images:
        raise RuntimeError(f"No usable frames found under {data_folder}")

    if return_numpy:
        return (
            convert_img_numpy(training_images),
            training_filenames,
            np.stack(training_labels, axis=0),
        )
    return training_images, training_filenames, training_labels


load_dual_hose_data = load_hose_data


def create_hose_augmentation():
    """Geometry applied jointly to image + heatmaps; no Fliplr / 90° / 180°."""
    import imgaug.augmenters as iaa

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    often = lambda aug: iaa.Sometimes(0.8, aug)

    return iaa.Sequential(
        [
            often(
                iaa.OneOf([
                    iaa.Multiply((0.7, 1.3)),
                    iaa.Add((-25, 25)),
                    iaa.LinearContrast((0.7, 1.3)),
                    iaa.GammaContrast((0.7, 1.4)),
                    iaa.pillike.Autocontrast(),
                ])
            ),
            sometimes(
                iaa.OneOf([
                    iaa.Sharpen(alpha=(0, 0.8), lightness=(0.8, 1.3)),
                    iaa.GaussianBlur(sigma=(0.0, 1.0)),
                    iaa.MotionBlur(k=(3, 5)),
                ])
            ),
            sometimes(
                iaa.OneOf([
                    iaa.Dropout(p=(0.01, 0.05)),
                    iaa.CoarseDropout(0.02, size_percent=(0.02, 0.08)),
                    iaa.ReplaceElementwise(0.01, [0, 255]),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.06 * 255)),
                    iaa.AdditivePoissonNoise(lam=(0, 8)),
                ])
            ),
            often(
                iaa.Affine(
                    rotate=(-15, 15),
                    scale={"x": (0.9, 1.08), "y": (0.9, 1.08)},
                    translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                    shear=(-4, 4),
                    order=1,
                    mode="constant",
                    cval=0,
                )
            ),
        ],
        random_order=False,
    )


def _HoseKeypointGeneratorBase():
    import tensorflow as tf

    class HoseKeypointGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            augmentation=None,
            batch_size: int = 32,
            training: bool = True,
            shuffle: bool = True,
        ):
            super().__init__()
            if images.shape[0] != labels.shape[0]:
                raise ValueError("images/labels length mismatch")
            if images.shape[1:3] != labels.shape[1:3]:
                raise ValueError("images/labels spatial mismatch")

            self.images = images
            self.labels = labels
            self.batch_size = batch_size
            self.training = training
            self.shuffle = shuffle
            self.seq = augmentation
            self.sample_size = int(images.shape[0])
            self.n_keypoints = int(labels.shape[3])
            self.indexes = np.arange(self.sample_size)
            self.on_epoch_end()

        def __len__(self):
            return int(np.floor(self.sample_size / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            return self._generate(indexes)

        def on_epoch_end(self):
            self.indexes = np.arange(self.sample_size)
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def _generate(self, indexes):
            from imgaug.augmentables.heatmaps import HeatmapsOnImage

            X = self.images[indexes].copy()
            y = self.labels[indexes].copy()

            if self.training and self.seq is not None:
                heatmaps = [
                    HeatmapsOnImage(yi.astype(np.float32) / 255.0, shape=xi.shape)
                    for xi, yi in zip(X, y)
                ]
                X, heatmaps = self.seq(images=X, heatmaps=heatmaps)
                y = np.stack(
                    [np.clip(h.get_arr(), 0.0, 1.0) for h in heatmaps], axis=0
                )
                X = np.asarray(X, dtype=np.float32)
            else:
                X = X.astype(np.float32)
                y = y.astype(np.float32)
                if float(np.max(y)) > 1.0:
                    y = y / 255.0

            return X, y.astype(np.float32)

    return HoseKeypointGenerator


_HoseKeypointGeneratorCls = None


def get_hose_generator_class():
    global _HoseKeypointGeneratorCls
    if _HoseKeypointGeneratorCls is None:
        _HoseKeypointGeneratorCls = _HoseKeypointGeneratorBase()
    return _HoseKeypointGeneratorCls


def dual_hose_loss(
    y_true,
    y_pred,
    pos_weight: float = 1000.0,
    channel_weights: Sequence[float] = (1.0, 1.0),
    exclusivity_weight: float = 50.0,
    coactivation_weight: float = 10.0,
    exclusive_pairs: Sequence[Tuple[int, int]] = EXCLUSIVE_PAIRS,
):
    """Weighted MSE + L↔R exclusivity + coactivation (no screen-order hinge)."""
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    n_ch = int(y_true.shape[-1])

    mse = 0.0
    weights = list(channel_weights) + [1.0] * max(0, n_ch - len(channel_weights))
    for c in range(n_ch):
        w = 1.0 + pos_weight * y_true[..., c]
        mse_c = tf.reduce_mean(w * tf.square(y_true[..., c] - y_pred[..., c]))
        mse = mse + float(weights[c]) * mse_c

    exclusivity = 0.0
    for a, b in exclusive_pairs:
        if a < n_ch and b < n_ch:
            exclusivity = exclusivity + tf.reduce_mean(y_pred[..., a] * y_true[..., b])
            exclusivity = exclusivity + tf.reduce_mean(y_pred[..., b] * y_true[..., a])

    coactivation = 0.0
    n_pairs = 0
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            coactivation = coactivation + tf.reduce_mean(y_pred[..., i] * y_pred[..., j])
            n_pairs += 1
    if n_pairs > 0:
        coactivation = coactivation / float(n_pairs)

    return (
        mse
        + exclusivity_weight * exclusivity
        + coactivation_weight * coactivation
    )


def _write_keypoint_csv(
    keypoints: VideoKeypoints,
    csv_path: Path,
    scale_height: float,
    scale_width: float,
    threshold: float,
    delimiter: str = ",",
) -> int:
    return keypoints.to_csv(
        str(csv_path),
        scale_height=scale_height,
        scale_width=scale_width,
        threshold=threshold,
        delimiter=delimiter,
    )


def _hard_peak_keypoint(heatmap: np.ndarray) -> Keypoint:
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heatmap.shape}")
    peak_map = np.zeros_like(heatmap)
    flat = int(np.argmax(heatmap))
    row, col = np.unravel_index(flat, heatmap.shape)
    peak_map[row, col] = heatmap[row, col]
    return Keypoint(peak_map)


def _extract_exclusive_peaks_multi(
    heats: np.ndarray,
    min_separation: float = 8.0,
) -> List[Keypoint]:
    if heats.ndim != 3:
        raise ValueError(f"Expected (H,W,C) heats, got {heats.shape}")
    n_ch = heats.shape[-1]
    winner = np.argmax(heats, axis=-1)
    exclusive = np.zeros_like(heats)
    for c in range(n_ch):
        mask = winner == c
        exclusive[..., c][mask] = heats[..., c][mask]

    peaks = [_hard_peak_keypoint(exclusive[..., c]) for c in range(n_ch)]

    order = sorted(range(n_ch), key=lambda c: peaks[c].prob, reverse=True)
    placed: List[int] = []
    for c in order:
        conflict = any(
            float(np.hypot(peaks[c].x - peaks[p].x, peaks[c].y - peaks[p].y))
            < min_separation
            for p in placed
        )
        if conflict:
            masked = heats[..., c].copy()
            yy, xx = np.ogrid[: masked.shape[0], : masked.shape[1]]
            for p in placed:
                disk = (yy - peaks[p].x) ** 2 + (xx - peaks[p].y) ** 2 <= min_separation ** 2
                masked[disk] = 0.0
            peaks[c] = _hard_peak_keypoint(masked)
        placed.append(c)

    return peaks


def _extract_exclusive_peaks(
    heat_l: np.ndarray,
    heat_r: np.ndarray,
    min_separation: float = 8.0,
) -> Tuple[Keypoint, Keypoint]:
    peaks = _extract_exclusive_peaks_multi(
        np.stack([heat_l, heat_r], axis=-1),
        min_separation=min_separation,
    )
    return peaks[0], peaks[1]


def process_video_batch_dual(
    video_paths: List[str],
    model,
    batch_size: int,
    threshold: float = 0.5,
    delimiter: str = ",",
    require_both: bool = True,
    log_file: Optional[str] = None,
    left_output_name: str = "hoseL_keypoints.csv",
    right_output_name: str = "hoseR_keypoints.csv",
) -> None:
    """Run 2-channel L/R hose inference; writes hoseL/hoseR CSVs next to each video."""
    if not video_paths:
        raise ValueError("video_paths is empty")

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(Path(video_paths[0]).parent / f"hose_lr_processing_{timestamp}.log")

    log_handle = open(log_file, "w")
    log_handle.write(f"Dual Hose (L/R) Processing Log - Started at {datetime.now()}\n")
    log_handle.write("=" * 80 + "\n\n")

    input_shape = model.input_shape
    model_input_height = input_shape[1]
    model_input_width = input_shape[2]
    model_input_channels = input_shape[3]

    total_videos = len(video_paths)
    total_frames_processed = 0
    total_processing_time = 0.0

    for video_idx, video_path in enumerate(video_paths, 1):
        video_path = str(video_path)
        log_handle.write(f"\nProcessing video {video_idx}/{total_videos}: {video_path}\n")
        log_handle.write("-" * 80 + "\n")

        vidcap = None
        progress = None

        try:
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            height, width = get_video_height_width(vidcap)
            height, width = int(height), int(width)
            num_frames = get_total_frames(vidcap)
            scale_height = height / model_input_height
            scale_width = width / model_input_width

            log_handle.write(f"Video dimensions: {width}x{height}\n")
            log_handle.write(f"Number of frames: {num_frames}\n")

            progress = tqdm(
                total=num_frames,
                desc=f"Video {video_idx}/{total_videos}",
                position=0,
                leave=True,
            )

            prediction_batch = np.zeros(
                (batch_size, height, width, model_input_channels),
                dtype=np.uint8,
            )
            batch_counter = 0
            frame_counter = 0

            left_kps = VideoKeypoints(model_input_height, model_input_width)
            right_kps = VideoKeypoints(model_input_height, model_input_width)

            start_time = time.time()
            while True:
                success, image = vidcap.read()
                if not success:
                    break

                prediction_batch[batch_counter] = image
                batch_counter += 1

                if batch_counter == batch_size:
                    labels = model.predict_on_batch(prediction_batch)
                    _add_dual_frames(
                        left_kps,
                        right_kps,
                        labels,
                        frame_counter,
                        threshold=threshold,
                        require_both=require_both,
                    )
                    frame_counter += batch_size
                    batch_counter = 0
                    progress.update(batch_size)

            if batch_counter > 0:
                labels = model.predict_on_batch(prediction_batch[:batch_counter])
                _add_dual_frames(
                    left_kps,
                    right_kps,
                    labels,
                    frame_counter,
                    threshold=threshold,
                    require_both=require_both,
                )
                progress.update(batch_counter)

            processing_time = time.time() - start_time
            total_processing_time += processing_time
            total_frames_processed += num_frames

            out_dir = Path(video_path).parent
            write_thresh = 0.0 if require_both else threshold
            n_left = _write_keypoint_csv(
                left_kps, out_dir / left_output_name,
                scale_height, scale_width, write_thresh, delimiter,
            )
            n_right = _write_keypoint_csv(
                right_kps, out_dir / right_output_name,
                scale_height, scale_width, write_thresh, delimiter,
            )

            log_handle.write(f"Processed {num_frames} frames in {processing_time:.2f}s\n")
            log_handle.write(f"Saved L={n_left} R={n_right}\n")

        except Exception as e:
            log_handle.write(f"ERROR: {e}\n{traceback.format_exc()}\n")
        finally:
            if vidcap is not None:
                vidcap.release()
            if progress is not None:
                progress.close()

    log_handle.write(f"\nCompleted at: {datetime.now()}\n")
    log_handle.close()
    print(f"\nProcessing complete. Log file: {log_file}")


def _add_dual_frames(
    left_kps: VideoKeypoints,
    right_kps: VideoKeypoints,
    labels: np.ndarray,
    frame_offset: int,
    threshold: float,
    require_both: bool,
) -> None:
    if labels.ndim != 4 or labels.shape[-1] < 2:
        raise ValueError(f"Expected labels (N,H,W,>=2), got {labels.shape}")

    for i in range(labels.shape[0]):
        left, right = _extract_exclusive_peaks(labels[i, :, :, 0], labels[i, :, :, 1])
        frame_num = i + frame_offset
        if require_both and (left.prob < threshold or right.prob < threshold):
            continue
        left_kps.keypoints.append(left)
        left_kps.frames.append(frame_num)
        right_kps.keypoints.append(right)
        right_kps.frames.append(frame_num)
