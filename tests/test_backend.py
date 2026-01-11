import unittest

from backend import analyze_series


class AnalyzeSeriesTests(unittest.TestCase):
    def assertConvergent(self, expr):
        result = analyze_series(expr)
        self.assertTrue(result.startswith("Convergent"), msg=result)

    def assertDivergent(self, expr):
        result = analyze_series(expr)
        self.assertTrue(result.startswith("Divergent"), msg=result)

    def assertInconclusive(self, expr):
        result = analyze_series(expr)
        self.assertTrue(result.startswith("Inconclusive"), msg=result)

    def test_basic_p_series(self):
        self.assertDivergent("1/n")
        self.assertConvergent("1/n^2")

    def test_logarithmic_series(self):
        self.assertDivergent("1/(n*log(n))")
        self.assertConvergent("1/(n*log(n)^2)")
        self.assertConvergent("1/(n*log(n)*log(log(n))^2)")
        self.assertDivergent("1/(n*log(n)*log(log(n)))")

    def test_ratio_test_cases(self):
        self.assertConvergent("1/2^n")
        self.assertConvergent("n/2^n")

    def test_limit_comparison(self):
        self.assertConvergent("arctan(n)/n^2")

    def test_alternating(self):
        result = analyze_series("(-1)^n/n")
        self.assertTrue(
            result.startswith("Convergent") and "Alternating" in result,
            msg=result,
        )
        result = analyze_series("(-1)^n/sqrt(n)")
        self.assertTrue(
            result.startswith("Convergent") and "Alternating" in result,
            msg=result,
        )

    def test_power_like_asymptotics(self):
        self.assertDivergent("sin(1/n)")
        self.assertDivergent("1/(n+sin(n))")
        self.assertDivergent("(n+1)/n^2")

    def test_dirichlet(self):
        result = analyze_series("sin(n)/n")
        self.assertTrue(
            result.startswith("Convergent") and "Dirichlet" in result,
            msg=result,
        )

    def test_log_over_n(self):
        self.assertDivergent("log(n)/n")
        result = analyze_series("(-1)^n*log(n)/n")
        self.assertTrue(
            result.startswith("Convergent") and "Alternating" in result,
            msg=result,
        )

    def test_nth_term_divergence(self):
        self.assertDivergent("(-1)^n")

    def test_invalid_expression(self):
        result = analyze_series("1/(")
        self.assertTrue(result.startswith("Invalid expression"), msg=result)


if __name__ == "__main__":
    unittest.main()
