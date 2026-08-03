import unittest
from pathlib import Path
import numpy as np
from scenario_reduction import generate_scenarios, reduce_scenarios

class ScenarioReductionTests(unittest.TestCase):
    def test_generate_scenarios_returns_requested_count(self) -> None:
        scenarios, probabilities = generate_scenarios(1000, seed=42)
        self.assertEqual(len(scenarios), 1000)
        self.assertEqual(len(probabilities), 1000)
        self.assertTrue(np.allclose(probabilities, 1.0 / 1000))
        self.assertEqual(scenarios.shape[1], 3)

        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / "example" / "input" / "scenarios.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = np.column_stack((scenarios, probabilities))
        np.savetxt(
            output_path,
            data_to_save,
            delimiter=",",
            header="scenario_1,scenario_2,scenario_3,probability",
            comments="",
            fmt="%.8f",
        )

        self.assertTrue(output_path.exists())

    def test_reduce_scenarios_reduces_to_target_count(self) -> None:
        scenarios, probabilities = generate_scenarios(1000, seed=42)
        reduced_scenarios, reduced_probabilities = reduce_scenarios(
            scenarios,
            probabilities,
            50,
        )
        self.assertEqual(len(reduced_scenarios), 50)
        self.assertEqual(len(reduced_probabilities), 50)
        self.assertTrue(np.isclose(reduced_probabilities.sum(), 1.0))


if __name__ == "__main__":
    unittest.main()
