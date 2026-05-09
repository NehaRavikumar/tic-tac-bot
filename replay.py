import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# --------------- Defaults ---------------
DEFAULT_PORT = "/dev/tty.usbmodem5AE60557941"
DEFAULT_ROBOT_ID = "follower-1"
# ----------------------------------------


def replay_episode(
    repo_id: str,
    episode_idx: int = 0,
    *,
    robot: SO101Follower | None = None,
    port: str = DEFAULT_PORT,
    robot_id: str = DEFAULT_ROBOT_ID,
    fps: int | None = None,
    say: bool = True,
) -> None:
    """Replay a recorded episode's actions on the SO-101 follower arm.

    Args:
        repo_id: Dataset repo id (e.g. "cadenli/tictactoe-point"). Loads from
            `$HF_LEROBOT_HOME/{repo_id}` locally if available, else from HF Hub.
        episode_idx: 0-based index of the episode to replay.
        robot: Pre-connected SO101Follower. When None, this function builds one
            and handles connect/disconnect itself.
        port: Follower USB port. Used only when `robot` is None.
        robot_id: Calibration id. Used only when `robot` is None.
        fps: Override playback rate. Defaults to the dataset's recorded fps.
        say: Vocalize a status message at the start of replay.
    """
    dataset = LeRobotDataset(repo_id, episodes=[episode_idx])
    actions = dataset.select_columns("action")
    action_names = dataset.features["action"]["names"]
    target_fps = fps if fps is not None else dataset.fps

    owns_robot = robot is None
    if owns_robot:
        robot = SO101Follower(
            SO101FollowerConfig(
                port=port,
                id=robot_id,
                disable_torque_on_disconnect=False,
            )
        )
        robot.connect()

    try:
        if say:
            log_say(f"Replaying episode {episode_idx}")

        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()

            action = {
                name: float(actions[idx]["action"][i])
                for i, name in enumerate(action_names)
            }
            robot.send_action(action)

            precise_sleep(max(1.0 / target_fps - (time.perf_counter() - t0), 0.0))
    finally:
        if owns_robot:
            try:
                robot.disconnect()
            except RuntimeError as e:
                print(f"Warning: robot disconnect error: {e}")


if __name__ == "__main__":
    print("=== SO-101 Replay ===\n")
    hf_user = input("HuggingFace username: ").strip()
    dataset_name = input("Dataset name: ").strip()
    episode_input = input("Episode index [0]: ").strip()
    episode_idx = int(episode_input) if episode_input else 0

    replay_episode(
        repo_id=f"{hf_user}/{dataset_name}",
        episode_idx=episode_idx,
    )
