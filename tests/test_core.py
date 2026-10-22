import unittest

from train.config import TrainConfig
from train.data import build_text, map_answer


class CoreTests(unittest.TestCase):
    def test_build_text_joins_context_and_question_without_separator(self):
        self.assertEqual(
            build_text("Paris is in France.", "Is Paris in France?"),
            "Paris is in France. Is Paris in France?",
        )

    def test_build_text_handles_empty_context(self):
        self.assertEqual(build_text("", "Is this true?"), "Is this true?")

    def test_build_text_handles_empty_question(self):
        self.assertEqual(build_text("A statement.", ""), "A statement.")

    def test_map_answer(self):
        self.assertEqual(map_answer(True), "yes")
        self.assertEqual(map_answer(False), "no")
        self.assertEqual(map_answer("no-answer"), "idk")

    def test_map_answer_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            map_answer("maybe")

    def test_label_order(self):
        config = TrainConfig()
        self.assertEqual(config.label2id, {"yes": 0, "no": 1, "idk": 2})
        self.assertEqual(config.id2label, {0: "yes", 1: "no", 2: "idk"})


if __name__ == "__main__":
    unittest.main()
