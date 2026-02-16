#!/usr/bin/env python3
"""Create hybrid frames by merging semantic segmentation with photo frames."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


ColorBGR = Tuple[int, int, int]
DEFAULT_CLASS_COLORS: Dict[str, ColorBGR] = {
    "Terrain": (128, 128, 128),
    "Unpaved Route": (0, 165, 255),
    "Paved Road": (0, 0, 0),
    "Tree Trunk": (102, 51, 0),
    "Tree Foliage": (0, 128, 0),
    "Rocks": (96, 96, 96),
    "Large Shrubs": (0, 255, 0),
    "Low Vegetation": (0, 255, 127),
    "Wire Fence": (192, 192, 192),
    "Sky": (255, 0, 0),
    "Person": (255, 127, 80),
    "Vehicle": (0, 0, 255),
    "Building": (0, 255, 255),
    "ignore": (10, 10, 10),
    "Misc": (128, 0, 128),
    "Water": (255, 255, 0),
    "Animal": (204, 255, 204),
}

DEFAULT_REAL_WORLD_CLASSES = (
    "Unpaved Route",
    "Paved Road",
    "Person",
    "Vehicle",
    "Animal",
)


@dataclass
class FrameProcessingResult:
    """Output of a single frame-processing run."""

    hybrid_img: np.ndarray
    hybrid_img_window: np.ndarray
    mask_real: np.ndarray


class SegmentationProcessor:
    """Process semantic frames and merge relevant regions with photo frames."""

    def __init__(
        self,
        res: Tuple[int, int] = (1920, 1080),
        class_seg_bgr: Dict[str, ColorBGR] | None = None,
        real_world_classes: Iterable[str] = DEFAULT_REAL_WORLD_CLASSES,
    ) -> None:
        self.class_seg_bgr = class_seg_bgr or DEFAULT_CLASS_COLORS.copy()
        self.width, self.height = res
        self.lut = self._create_lut(real_world_classes)

    def _create_lut(self, real_world_classes: Iterable[str]) -> np.ndarray:
        """Build RGB->class lookup table for fast mask creation."""
        lut = np.zeros((256 * 256 * 256,), dtype=np.uint8)

        for class_name in real_world_classes:
            if class_name not in self.class_seg_bgr:
                raise KeyError(f"Unknown class in LUT definition: '{class_name}'")

            bgr = self.class_seg_bgr[class_name]
            lut[65536 * bgr[2] + 256 * bgr[1] + bgr[0]] = 1

        return lut

    @staticmethod
    def _bgr_image_to_rgb24_index(image: np.ndarray) -> np.ndarray:
        """Encode BGR image to flat RGB24 indices used by the LUT."""
        return (
            65536 * np.uint32(image[:, :, 2])
            + 256 * np.uint32(image[:, :, 1])
            + np.uint32(image[:, :, 0])
        )

    def recolor_connected_components(
        self,
        image: np.ndarray,
        target_color_bgr: ColorBGR,
        source_color_bgr: ColorBGR,
    ) -> np.ndarray:
        """
        Recolor source components only when they touch the target class.

        This is used to absorb adjacent classes (for example "Rocks")
        into a route class (for example "Unpaved Route").
        """
        target_mask = np.all(image == target_color_bgr, axis=-1).astype(np.uint8) * 255
        source_mask = np.all(image == source_color_bgr, axis=-1).astype(np.uint8) * 255

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            source_mask, connectivity=8, ltype=cv2.CV_32S
        )
        source_to_recolor_mask = np.zeros_like(source_mask, dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)

        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            dilated_component = cv2.dilate(component_mask, kernel, iterations=1)
            overlap = cv2.bitwise_and(dilated_component, target_mask)
            if cv2.countNonZero(overlap) > 0:
                source_to_recolor_mask = cv2.bitwise_or(
                    source_to_recolor_mask, component_mask
                )

        final_mask = cv2.bitwise_or(target_mask, source_to_recolor_mask)
        modified_image = image.copy()
        modified_image[final_mask > 0] = target_color_bgr
        return modified_image

    def process_frame(
        self, photo_img: np.ndarray, sem_img: np.ndarray
    ) -> FrameProcessingResult:
        """Create merged outputs for one frame pair."""
        if photo_img.shape[:2] != sem_img.shape[:2]:
            raise ValueError(
                "photo and semantic frames must have the same resolution, got "
                f"{photo_img.shape[:2]} and {sem_img.shape[:2]}"
            )

        frame_height, frame_width = sem_img.shape[:2]
        if (frame_width, frame_height) != (self.width, self.height):
            LOGGER.debug(
                "Resolution changed from %sx%s to %sx%s.",
                self.width,
                self.height,
                frame_width,
                frame_height,
            )
            self.width, self.height = frame_width, frame_height

        start_total = time.perf_counter()

        img_prior = self.lut[self._bgr_image_to_rgb24_index(sem_img)]
        mask_real = img_prior != 0
        LOGGER.debug("LUT and initial mask ready in %.4fs", time.perf_counter() - start_total)

        unpaved_route_color = self.class_seg_bgr["Unpaved Route"]
        rocks_color = self.class_seg_bgr["Rocks"]
        modified_seg = self.recolor_connected_components(
            sem_img,
            target_color_bgr=unpaved_route_color,
            source_color_bgr=rocks_color,
        )
        mask_new = np.all(modified_seg == unpaved_route_color, axis=2)
        mask_real = np.logical_or(mask_real, mask_new)
        LOGGER.debug(
            "Connected-component recolor ready in %.4fs",
            time.perf_counter() - start_total,
        )

        hybrid_img = sem_img.copy()
        hybrid_img[mask_real] = photo_img[mask_real]

        y_min, x_min, x_max = self._find_window_coords(img_prior)
        hybrid_img_window = sem_img.copy()
        hybrid_img_window[y_min:self.height, x_min : x_max + 1] = photo_img[
            y_min:self.height, x_min : x_max + 1
        ]

        LOGGER.debug(
            "Frame processed in %.4fs | mask coverage: %.2f%%",
            time.perf_counter() - start_total,
            float(mask_real.mean() * 100.0),
        )
        return FrameProcessingResult(hybrid_img, hybrid_img_window, mask_real)

    def _find_window_coords(self, img_prior: np.ndarray) -> Tuple[int, int, int]:
        """Find a bounding window where priority mask is present."""
        rows_with_data = np.where(np.any(img_prior != 0, axis=1))[0]
        cols_with_data = np.where(np.any(img_prior != 0, axis=0))[0]

        if rows_with_data.size == 0 or cols_with_data.size == 0:
            return 0, 0, self.width - 1

        y_min = int(rows_with_data[0])
        x_min = int(cols_with_data[0])
        x_max = int(cols_with_data[-1])
        return y_min, x_min, x_max

    def process_video(
        self,
        semantic_video_path: Path,
        photo_video_path: Path,
        hybrid_output_path: Path,
        mask_output_path: Path,
        codec: str = "FFV1",
        show_preview: bool = False,
    ) -> None:
        """Process two synchronized videos and write hybrid and mask outputs."""
        semantic_cap = cv2.VideoCapture(str(semantic_video_path))
        photo_cap = cv2.VideoCapture(str(photo_video_path))

        if not semantic_cap.isOpened():
            raise FileNotFoundError(f"Cannot open semantic video: {semantic_video_path}")
        if not photo_cap.isOpened():
            raise FileNotFoundError(f"Cannot open photo video: {photo_video_path}")

        width = int(semantic_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        height = int(semantic_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        fps = semantic_cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        self.width, self.height = width, height

        fourcc = cv2.VideoWriter_fourcc(*codec)
        hybrid_writer = cv2.VideoWriter(str(hybrid_output_path), fourcc, fps, (width, height))
        mask_writer = cv2.VideoWriter(
            str(mask_output_path), fourcc, fps, (width, height), isColor=False
        )

        if not hybrid_writer.isOpened():
            raise RuntimeError(f"Cannot open output writer: {hybrid_output_path}")
        if not mask_writer.isOpened():
            raise RuntimeError(f"Cannot open output writer: {mask_output_path}")

        LOGGER.info("Starting video processing...")
        LOGGER.info("Semantic: %s", semantic_video_path)
        LOGGER.info("Photo   : %s", photo_video_path)
        LOGGER.info("Output  : %s, %s", hybrid_output_path, mask_output_path)

        frame_index = 0
        start = time.perf_counter()
        try:
            while True:
                ret_photo, photo_img = photo_cap.read()
                ret_sem, sem_img = semantic_cap.read()

                if not ret_photo or not ret_sem:
                    break

                result = self.process_frame(photo_img, sem_img)
                mask_image = result.mask_real.astype(np.uint8) * 255
                hybrid_writer.write(result.hybrid_img)
                mask_writer.write(mask_image)

                frame_index += 1
                if frame_index % 30 == 0:
                    elapsed = time.perf_counter() - start
                    fps_now = frame_index / elapsed if elapsed > 0 else 0.0
                    LOGGER.info("Processed %d frames | avg %.2f FPS", frame_index, fps_now)

                if show_preview:
                    cv2.imshow("hybrid_img", result.hybrid_img)
                    cv2.imshow("mask", mask_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        LOGGER.info("Preview interrupted by user (q).")
                        break
        finally:
            semantic_cap.release()
            photo_cap.release()
            hybrid_writer.release()
            mask_writer.release()
            cv2.destroyAllWindows()

        elapsed_total = time.perf_counter() - start
        avg_fps = frame_index / elapsed_total if elapsed_total > 0 else 0.0
        LOGGER.info(
            "Finished. Frames: %d | Total time: %.2fs | Avg FPS: %.2f",
            frame_index,
            elapsed_total,
            avg_fps,
        )


def parse_args() -> argparse.Namespace:
    """CLI arguments for running the processor as a standalone script."""
    parser = argparse.ArgumentParser(
        description="Merge semantic segmentation and photo videos into hybrid outputs."
    )
    parser.add_argument("--semantic", default="semantic.avi", type=Path)
    parser.add_argument("--photo", default="photo.avi", type=Path)
    parser.add_argument("--hybrid-out", default="hybrid.avi", type=Path)
    parser.add_argument("--mask-out", default="mask.avi", type=Path)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity of status logs.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show live preview windows during processing.",
    )
    return parser.parse_args()


def main() -> None:
    """Script entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    processor = SegmentationProcessor()
    processor.process_video(
        semantic_video_path=args.semantic,
        photo_video_path=args.photo,
        hybrid_output_path=args.hybrid_out,
        mask_output_path=args.mask_out,
        show_preview=args.show_preview,
    )


if __name__ == "__main__":
    main()
