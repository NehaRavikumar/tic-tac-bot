from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from .board import Mark, MoveResult, next_robot_move, normalize_board


DEFAULT_MODEL = "gpt-4.1-mini"


def parse_board_image(image_path: str | Path, model: str = DEFAULT_MODEL) -> tuple[tuple[str, str, str], ...]:
    """Use a vision-capable LLM to parse a tic-tac-toe board image.

    Returns a 3x3 board where each cell is "X", "O", or "".
    """
    from openai import OpenAI

    image_url = _image_to_data_url(Path(image_path))
    client = OpenAI()

    response = client.responses.create(
        model=model,
        text={
            "format": {
                "type": "json_schema",
                "name": "tic_tac_toe_board",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "board": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {"type": "string", "enum": ["X", "O", ""]},
                            },
                        }
                    },
                    "required": ["board"],
                },
            }
        },
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Parse this tic-tac-toe board image. Return only JSON with a "
                            "'board' key. The board must be a 3x3 array from top row to "
                            "bottom row, left to right. Use 'X' for X marks, 'O' for O "
                            "marks, and '' for empty cells. Do not infer a mark unless it "
                            "is visible in the corresponding cell."
                        ),
                    },
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            }
        ],
    )

    payload = _parse_json_response(response)
    board = payload.get("board")
    if not isinstance(board, list):
        raise ValueError("LLM response did not include a 'board' array.")
    return normalize_board(board)


def next_robot_move_from_image(
    image_path: str | Path,
    robot_mark: Mark | None = None,
    model: str = DEFAULT_MODEL,
) -> MoveResult:
    board = parse_board_image(image_path, model=model)
    return next_robot_move(board, robot_mark=robot_mark)


def _image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_json_response(response: Any) -> dict[str, Any]:
    text = getattr(response, "output_text", None)
    if not text:
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                candidate = getattr(content, "text", None)
                if candidate:
                    parts.append(candidate)
        text = "\n".join(parts)

    if not text:
        raise ValueError("LLM response did not contain text output.")

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object.")
    return payload
