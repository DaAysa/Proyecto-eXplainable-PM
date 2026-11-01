import unittest

from promoai.model_generation.code_extraction import execute_code_and_get_variable


class ExecuteCodeTests(unittest.TestCase):
    def test_comprehension_can_access_name_assigned_by_executed_code(self):
        code = """
col_names = ["O_Accepted", "other"]
response_cols = [
    column
    for column in ["O_Accepted", "O_Refused", "O_Cancelled"]
    if column in col_names
]
final_value = response_cols
"""

        result = execute_code_and_get_variable(code, "final_value")

        self.assertEqual(result, ["O_Accepted"])

    def test_comprehension_can_access_supplied_namespace(self):
        code = "final_value = [value * factor for value in values]"

        result = execute_code_and_get_variable(
            code,
            "final_value",
            namespace={"values": [1, 2, 3], "factor": 2},
        )

        self.assertEqual(result, [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
