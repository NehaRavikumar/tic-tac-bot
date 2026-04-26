import unittest

from tictac_bot.board import InvalidBoardError, best_move, next_robot_move, normalize_board


class StrategyTests(unittest.TestCase):
    def test_o_takes_center_when_available(self) -> None:
        board = normalize_board(
            [
                ["X", "", ""],
                ["", "", ""],
                ["", "", ""],
            ]
        )

        self.assertEqual(best_move(board, "O"), (1, 1))

    def test_o_wins_immediately(self) -> None:
        board = normalize_board(
            [
                ["X", "X", ""],
                ["O", "O", ""],
                ["X", "", ""],
            ]
        )

        self.assertEqual(best_move(board, "O"), (2, 1))

    def test_o_blocks_immediate_x_win(self) -> None:
        board = normalize_board(
            [
                ["X", "X", ""],
                ["", "O", ""],
                ["", "", ""],
            ]
        )

        self.assertEqual(best_move(board, "O"), (2, 0))

    def test_coordinates_are_x_y_from_top_left(self) -> None:
        result = next_robot_move(
            [
                ["X", "", ""],
                ["", "", ""],
                ["", "", ""],
            ],
            robot_mark="O",
        )

        self.assertEqual(result.coordinate, (1, 1))

    def test_rejects_wrong_turn_for_robot(self) -> None:
        board = normalize_board(
            [
                ["X", "O", ""],
                ["", "", ""],
                ["", "", ""],
            ]
        )

        with self.assertRaises(InvalidBoardError):
            best_move(board, "O")


if __name__ == "__main__":
    unittest.main()
