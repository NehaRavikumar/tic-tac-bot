from __future__ import annotations

import argparse
import json

from .vision import DEFAULT_MODEL, next_robot_move_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose the robot's next tic-tac-toe move from a board image.")
    parser.add_argument("image", help="Path to an image of the tic-tac-toe board.")
    parser.add_argument(
        "--robot",
        choices=("X", "O", "auto"),
        default="auto",
        help="Robot's mark. Defaults to auto, which infers whose turn it is.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Vision model to call. Defaults to {DEFAULT_MODEL}.")
    args = parser.parse_args()

    robot_mark = None if args.robot == "auto" else args.robot
    result = next_robot_move_from_image(args.image, robot_mark=robot_mark, model=args.model)
    print(
        json.dumps(
            {
                "move": {"x": result.x, "y": result.y},
                "robot": result.robot_mark,
                "board": result.board,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
