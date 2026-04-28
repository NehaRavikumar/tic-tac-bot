import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import cv2

# Load .env files before importing anything that needs the API key.
_PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    for _candidate in (
        _PROJECT_ROOT / ".env.local",
        _PROJECT_ROOT / ".env",
        _PROJECT_ROOT / "tictac_bot" / ".env.local",
        _PROJECT_ROOT / "tictac_bot" / ".env",
    ):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
except ImportError:
    # python-dotenv is optional; without it the user must export env vars themselves.
    pass

from replay import DEFAULT_PORT, DEFAULT_ROBOT_ID, replay_episode
from tictac_bot.board import (
    InvalidBoardError,
    Mark,
    is_full,
    next_player,
    next_robot_move,
    winners,
)
from tictac_bot.vision import parse_board_image

if TYPE_CHECKING:
    from lerobot.robots.so_follower import SO101Follower

REPO_ID = "cadenli/tictactoe-draw"  # cadenli/tictactoe-point
CAM_REPO_ID = "cadenli/tictac-cam"

CAPTURE_DIR = Path(__file__).resolve().parent / "captures"
CAPTURE_CAMERA_INDEX = 0
CAPTURE_WARMUP_FRAMES = 5

# Grid layout of episode indices on the physical 3x3 board:
#   0 1 2
#   3 4 5
#   6 7 8
# Indexed as coords[row][col], so episode = row * 3 + col.
# Special case: (-1, -1) replays the camera-view dataset, then snaps a photo
# from camera index 0 and saves it under captures/.


def coords_to_episode(row: int, col: int) -> int:
    if not (0 <= row <= 2 and 0 <= col <= 2):
        raise ValueError(f"row and col must be in [0, 2], got ({row}, {col})")
    return row * 3 + col


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


@contextmanager
def connected_robot(
    port: str = DEFAULT_PORT,
    robot_id: str = DEFAULT_ROBOT_ID,
) -> Iterator["SO101Follower"]:
    """Connect once and reuse a single SO101 follower across multiple replays."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(
        SO101FollowerConfig(
            port=port,
            id=robot_id,
            disable_torque_on_disconnect=False,
        )
    )
    robot.connect()
    try:
        yield robot
    finally:
        try:
            robot.disconnect()
        except RuntimeError as e:
            print(f"Warning: robot disconnect error: {e}")


def go_to_camera_view(robot: "SO101Follower | None" = None) -> None:
    """Replay the camera-view dataset so the arm clears the board."""
    replay_episode(repo_id=CAM_REPO_ID, episode_idx=0, robot=robot, say=False)


def draw_at(row: int, col: int, robot: "SO101Follower | None" = None) -> None:
    """Replay the draw episode for cell (row, col)."""
    episode_idx = coords_to_episode(row, col)
    replay_episode(repo_id=REPO_ID, episode_idx=episode_idx, robot=robot, say=False)


def play(row: int, col: int) -> None:
    """Manual single-cell mode (kept for backward compatibility)."""
    if row == -1 and col == -1:
        print(f"Playing camera view -> {CAM_REPO_ID} episode 0")
        go_to_camera_view()
        out_path = capture_frame()
        print(f"Saved camera capture -> {out_path}")
        return

    print(f"Playing [{row}][{col}] -> episode {coords_to_episode(row, col)}")
    draw_at(row, col)


def _format_board(board) -> str:
    rendered = ["  " + " | ".join(c if c else "." for c in row) for row in board]
    return "\n  ---+---+---\n".join(rendered)


def play_game(
    robot_mark: Mark | None = None,
    *,
    max_rounds: int = 9,
) -> None:
    """Run the full perceive-plan-act loop until the game ends.

    Each round:
      1. Replay CAM_REPO_ID so the arm clears the board.
      2. Capture a frame and ask the vision LLM what's on the board.
      3. If somebody won or the board is full, stop.
      4. If it isn't the robot's turn yet, wait for the human and re-capture.
      5. Otherwise replay the draw episode for the chosen cell, then wait for
         the human's next move.
    """
    chosen_mark: Mark | None = robot_mark

    with connected_robot() as robot:
        for round_idx in range(1, max_rounds + 1):
            print(f"\n=== Round {round_idx} ===")

            print(f"Moving arm to camera-view pose ({CAM_REPO_ID} episode 0)...")
            go_to_camera_view(robot=robot)

            print("Capturing board image...")
            image_path = capture_frame()
            print(f"Saved -> {image_path}")

            try:
                board = parse_board_image(image_path)
            except Exception as e:
                print(f"Vision parse failed: {e}")
                if input("Retry capture? [Y/n]: ").strip().lower() == "n":
                    break
                continue

            print("Parsed board:")
            print(_format_board(board))

            won_by = winners(board)
            if won_by:
                print(f"\nGame over: {next(iter(won_by))} wins.")
                break
            if is_full(board):
                print("\nGame over: draw.")
                break

            if chosen_mark is None:
                chosen_mark = next_player(board)
                print(f"Robot will play as {chosen_mark}.")

            try:
                move = next_robot_move(board, robot_mark=chosen_mark)
            except InvalidBoardError as e:
                # Most often: not the robot's turn yet because the human
                # hasn't placed their mark on the physical board.
                print(f"Robot can't move from this board yet: {e}")
                if input("Wait for human, then retry? [Y/n]: ").strip().lower() == "n":
                    break
                continue

            row, col = move.y, move.x
            episode_idx = coords_to_episode(row, col)
            print(
                f"Robot ({move.robot_mark}) plays row={row}, col={col} "
                f"-> episode {episode_idx}"
            )
            draw_at(row, col, robot=robot)

            resp = input(
                "Robot done. Press Enter after the human moves (or 'q' to quit): "
            ).strip().lower()
            if resp == "q":
                print("Quitting game loop.")
                break
        else:
            print("\nReached max rounds without game ending.")


if __name__ == "__main__":
    print("=== Tic-Tac-Toe Robot ===")
    print("Commands:")
    print("  game           play a full game (default)")
    print("  <row> <col>    move to one cell (0-2)")
    print("  -1 -1          move to camera view + capture")
    cmd = input("Command [game]: ").strip()

    if cmd == "" or cmd.lower() in {"g", "game"}:
        mark_in = (input("Robot mark X/O/auto [auto]: ").strip().upper() or "AUTO")
        if mark_in not in {"X", "O", "AUTO"}:
            raise SystemExit("Mark must be X, O, or auto.")
        robot_mark: Mark | None = None if mark_in == "AUTO" else mark_in  # type: ignore[assignment]
        play_game(robot_mark=robot_mark)
    else:
        parts = cmd.split()
        if len(parts) != 2:
            raise SystemExit(
                "Please enter two integers separated by a space, or 'game' for full game mode."
            )
        row_in, col_in = int(parts[0]), int(parts[1])
        play(row_in, col_in)
