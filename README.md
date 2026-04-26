# Tic Tac Bot

Given an image of a tic-tac-toe board, this package asks a vision-capable LLM
to identify X/O placements, then returns the optimal next robot move as an
`(x, y)` coordinate where the top-left square is `(0, 0)`.

By default the robot mark is inferred from whose turn it is. Pass `--robot O`
or `--robot X` when the physical robot has a fixed mark.

## Setup

```bash
pip install -e .
export OPENAI_API_KEY=...
```

## CLI

```bash
tic-tac-bot path/to/board.jpg
```

Example output:

```json
{"move":{"x":1,"y":1},"robot":"O","board":[["X","",""],["","O",""],["","","X"]]}
```

## Python

```python
from tictac_bot import next_robot_move_from_image

move = next_robot_move_from_image("board.jpg", robot_mark="O")
print(move.coordinate)  # (x, y)
```
