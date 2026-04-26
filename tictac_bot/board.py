from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Sequence

Cell = Literal["X", "O", ""]
Mark = Literal["X", "O"]
Coordinate = tuple[int, int]
Board = tuple[tuple[Cell, Cell, Cell], tuple[Cell, Cell, Cell], tuple[Cell, Cell, Cell]]


WIN_LINES: tuple[tuple[Coordinate, Coordinate, Coordinate], ...] = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((2, 0), (1, 1), (0, 2)),
)


class InvalidBoardError(ValueError):
    pass


@dataclass(frozen=True)
class MoveResult:
    x: int
    y: int
    robot_mark: Mark
    board: Board

    @property
    def coordinate(self) -> Coordinate:
        return (self.x, self.y)


def normalize_board(raw_board: Sequence[Sequence[str]]) -> Board:
    if len(raw_board) != 3:
        raise InvalidBoardError("Board must have exactly 3 rows.")

    rows: list[tuple[Cell, Cell, Cell]] = []
    for raw_row in raw_board:
        if len(raw_row) != 3:
            raise InvalidBoardError("Each board row must have exactly 3 columns.")

        row: list[Cell] = []
        for raw_cell in raw_row:
            cell = str(raw_cell).strip().upper()
            if cell in {".", "-", "EMPTY", "NONE", "NULL"}:
                cell = ""
            if cell not in {"X", "O", ""}:
                raise InvalidBoardError(f"Invalid cell value: {raw_cell!r}.")
            row.append(cell)  # type: ignore[arg-type]
        rows.append(tuple(row))  # type: ignore[arg-type]

    board: Board = tuple(rows)  # type: ignore[assignment]
    validate_reachable_board(board)
    return board


def validate_reachable_board(board: Board) -> None:
    x_count = count_mark(board, "X")
    o_count = count_mark(board, "O")
    if o_count > x_count or x_count - o_count > 1:
        raise InvalidBoardError(
            f"Unreachable turn counts: X has {x_count}, O has {o_count}."
        )

    won_by = winners(board)
    x_won = "X" in won_by
    o_won = "O" in won_by
    if x_won and o_won:
        raise InvalidBoardError("Both players cannot have winning lines.")
    if x_won and x_count != o_count + 1:
        raise InvalidBoardError("X cannot win unless X has exactly one extra move.")
    if o_won and x_count != o_count:
        raise InvalidBoardError("O cannot win unless X and O have equal moves.")


def count_mark(board: Board, mark: Mark) -> int:
    return sum(cell == mark for row in board for cell in row)


def empty_cells(board: Board) -> tuple[Coordinate, ...]:
    return tuple((x, y) for y, row in enumerate(board) for x, cell in enumerate(row) if cell == "")


def winner(board: Board) -> Mark | None:
    won_by = winners(board)
    return next(iter(won_by)) if len(won_by) == 1 else None


def winners(board: Board) -> frozenset[Mark]:
    won_by: set[Mark] = set()
    for line in WIN_LINES:
        marks = [board[y][x] for x, y in line]
        if marks[0] and marks[0] == marks[1] == marks[2]:
            won_by.add(marks[0])  # type: ignore[arg-type]
    return frozenset(won_by)


def is_full(board: Board) -> bool:
    return not any(cell == "" for row in board for cell in row)


def next_player(board: Board) -> Mark:
    return "X" if count_mark(board, "X") == count_mark(board, "O") else "O"


def other_mark(mark: Mark) -> Mark:
    return "O" if mark == "X" else "X"


def place_mark(board: Board, coord: Coordinate, mark: Mark) -> Board:
    x, y = coord
    if not (0 <= x <= 2 and 0 <= y <= 2):
        raise InvalidBoardError(f"Coordinate out of range: {coord!r}.")
    if board[y][x] != "":
        raise InvalidBoardError(f"Cell {coord!r} is already occupied.")

    rows = [list(row) for row in board]
    rows[y][x] = mark
    return tuple(tuple(row) for row in rows)  # type: ignore[return-value]


def best_move(board: Board, robot_mark: Mark | None = None) -> Coordinate | None:
    if winners(board) or is_full(board):
        return None

    current = next_player(board)
    robot_mark = robot_mark or current
    if current != robot_mark:
        raise InvalidBoardError(
            f"It is {current}'s turn from this board, but robot is configured as {robot_mark}."
        )

    ranked_moves = sorted(empty_cells(board), key=move_preference)
    scored_moves = (
        (_minimax(place_mark(board, coord, robot_mark), other_mark(robot_mark), robot_mark, 1), coord)
        for coord in ranked_moves
    )
    return max(scored_moves, key=lambda item: (item[0], move_preference(item[1])))[1]


def next_robot_move(raw_board: Sequence[Sequence[str]], robot_mark: Mark | None = None) -> MoveResult:
    board = normalize_board(raw_board)
    move = best_move(board, robot_mark)
    if move is None:
        raise InvalidBoardError("No legal robot move is available.")
    x, y = move
    return MoveResult(x=x, y=y, robot_mark=robot_mark or next_player(board), board=board)


def move_preference(coord: Coordinate) -> int:
    if coord == (1, 1):
        return 4
    if coord in {(0, 0), (2, 0), (0, 2), (2, 2)}:
        return 3
    return 2


@lru_cache(maxsize=None)
def _minimax(board: Board, current_mark: Mark, robot_mark: Mark, depth: int) -> int:
    won_by = winner(board)
    if won_by == robot_mark:
        return 10 - depth
    if won_by == other_mark(robot_mark):
        return depth - 10
    if is_full(board):
        return 0

    scores = [
        _minimax(place_mark(board, coord, current_mark), other_mark(current_mark), robot_mark, depth + 1)
        for coord in empty_cells(board)
    ]
    return max(scores) if current_mark == robot_mark else min(scores)
