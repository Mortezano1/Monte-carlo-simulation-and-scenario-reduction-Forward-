import unittest

import numpy as np

from scenario_reduction import generate_scenarios, reduce_scenarios


class ScenarioReductionTests(unittest.TestCase):
    def test_generate_scenarios_returns_requested_count(self) -> None:
        scenarios, probabilities = generate_scenarios(1000, seed=42)

        self.assertEqual(len(scenarios), 1000)
        self.assertEqual(len(probabilities), 1000)
        self.assertTrue(np.allclose(probabilities, 1.0 / 1000))
        self.assertEqual(scenarios.shape[1], 3)

    def test_reduce_scenarios_reduces_to_target_count(self) -> None:
        scenarios, probabilities = generate_scenarios(1000, seed=42)
        reduced_scenarios, reduced_probabilities = reduce_scenarios(scenarios, probabilities, 50)

        self.assertEqual(len(reduced_scenarios), 50)
        self.assertEqual(len(reduced_probabilities), 50)
        self.assertTrue(np.isclose(reduced_probabilities.sum(), 1.0))


if __name__ == "__main__":
    unittest.main()
