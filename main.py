import time
from pathlib import Path

import cv2

from replay import replay_episode

REPO_ID = "cadenli/tictactoe-point"
CAM_REPO_ID = "cadenli/tictac-cam"

CAPTURE_DIR = Path(__file__).resolve().parent / "captures"
CAPTURE_CAMERA_INDEX = 0
CAPTURE_WARMUP_FRAMES = 5

# Grid layout of episode indices on the physical 3x3 board:
#   2 1 0
#   5 4 3
#   8 7 6
# Indexed as coords[row][col], so episode = row * 3 + (2 - col).
# Special case: (-1, -1) replays the camera-view dataset, then snaps a photo
# from camera index 0 and saves it under captures/.


def coords_to_episode(row: int, col: int) -> int:
    if not (0 <= row <= 2 and 0 <= col <= 2):
        raise ValueError(f"row and col must be in [0, 2], got ({row}, {col})")
    return row * 3 + (2 - col)


def capture_frame(camera_index: int = CAPTURE_CAMERA_INDEX) -> Path:
    """Grab one frame from `camera_index` and save it under captures/."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    try:
        # Discard a few warmup frames so auto-exposure/white-balance settle.
        frame = None
        for _ in range(CAPTURE_WARMUP_FRAMES):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame from camera {camera_index}")
        assert frame is not None

        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = CAPTURE_DIR / f"cam{camera_index}_{timestamp}.png"
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f"Failed to write image to {out_path}")
        return out_path
    finally:
        cap.release()


def play(row: int, col: int) -> None:
    if row == -1 and col == -1:
        print(f"Playing camera view -> {CAM_REPO_ID} episode 0")
        replay_episode(repo_id=CAM_REPO_ID, episode_idx=0)
        out_path = capture_frame()
        print(f"Saved camera capture -> {out_path}")
        return

    episode_idx = coords_to_episode(row, col)
    print(f"Playing [{row}][{col}] -> episode {episode_idx}")
    replay_episode(repo_id=REPO_ID, episode_idx=episode_idx)


if __name__ == "__main__":
    print("=== Tic-Tac-Toe Point ===")
    raw = input("Enter coords as 'row col' (0-2, or -1 -1 for camera view): ").strip().split()
    if len(raw) != 2:
        raise SystemExit("Please enter two integers separated by a space.")
    row, col = int(raw[0]), int(raw[1])
    play(row, col)
