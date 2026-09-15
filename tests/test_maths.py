import argparse

import torch

import ptychi.maths as pmath
import test_utils as tutils


class TestMaths(tutils.BaseTester):

    def test_trim_mean_matches_matlab_reference_cases(self):
        torch.testing.assert_close(
            pmath.trim_mean(torch.tensor([1.0, 2.0, 30.0]), 0.1),
            torch.tensor(11.0),
        )
        tied_values = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
        )
        torch.testing.assert_close(
            pmath.trim_mean(tied_values, 0.1),
            torch.tensor(12.1),
        )
        torch.testing.assert_close(
            pmath.trim_mean(tied_values, 0.2),
            torch.tensor(2.625),
        )
        torch.testing.assert_close(
            pmath.trim_mean(torch.tensor([1.0, 2.0, 3.0, torch.nan, 5.0]), 0.1),
            torch.tensor(2.75),
        )

    def test_trim_mean_uses_matlab_half_down_rounding(self):
        ten_values = torch.tensor([0.0] * 9 + [100.0])
        eleven_values = torch.tensor([0.0] * 10 + [100.0])
        twenty_values = torch.tensor(list(range(1, 20)) + [1000.0])

        torch.testing.assert_close(pmath.trim_mean(ten_values, 0.1), torch.tensor(10.0))
        torch.testing.assert_close(pmath.trim_mean(eleven_values, 0.1), torch.tensor(0.0))
        torch.testing.assert_close(pmath.trim_mean(twenty_values, 0.1), torch.tensor(10.5))

    def test_trim_mean_handles_nan_per_reduced_slice(self):
        x = torch.tensor([[1.0, torch.nan], [2.0, torch.nan], [3.0, torch.nan]])

        actual = pmath.trim_mean(x, 0.1, dim=0, keepdim=True)

        torch.testing.assert_close(
            actual,
            torch.tensor([[2.0, torch.nan]]),
            equal_nan=True,
        )
        torch.testing.assert_close(
            pmath.trim_mean(x[:2], 1.0, dim=0),
            torch.tensor([torch.nan, torch.nan]),
            equal_nan=True,
        )

    def test_trim_mean_supports_multiple_dimensions(self):
        x = torch.arange(24.0).reshape(2, 3, 4)

        actual = pmath.trim_mean(x, 0.1, dim=(0, 2), keepdim=True)

        torch.testing.assert_close(actual, x.mean(dim=(0, 2), keepdim=True))

    def test_project_zero_denominator(self):
        a = torch.tensor([[1 + 2j, 2], [3 + 4j, 4]], dtype=torch.complex128)
        b = torch.tensor([[0.0, 0.0], [1e-100, 0.0]], dtype=torch.complex128)

        projected = pmath.project(a, b, dim=-1)

        torch.testing.assert_close(projected[0], torch.zeros(2, dtype=torch.complex128))
        torch.testing.assert_close(
            projected[1], torch.tensor([3 + 4j, 0], dtype=torch.complex128)
        )
    
    def test_orthgonalize_gs(self):
        torch.manual_seed(123)
        x = torch.rand(2, 3, 4, 5)
        x = pmath.orthogonalize_gs(x, dim=(-2, -1), group_dim=1)
        prod = torch.sum(x[0, 0] * x[0, 1])
        assert prod < 1e-4


    def test_orthgonalize_svd(self):
        torch.manual_seed(123)
        x = torch.rand(2, 4, 5) + 1j * torch.rand(2, 4, 5)
        x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=0)
        prod = torch.sum(x[0] * x[1].conj())
        assert prod.abs() < 1e-4

        x = torch.rand(2, 3, 4, 5) + 1j * torch.rand(2, 3, 4, 5)
        x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=1)
        prod = torch.sum(x[0, 0] * x[0, 1].conj())
        assert prod.abs() < 1e-4

        x = torch.rand(2, 3, 4, 5) + 1j * torch.rand(2, 3, 4, 5)
        x = pmath.orthogonalize_svd(x, dim=(-2, -1), group_dim=0)
        prod = torch.sum(x[0, 0] * x[1, 0].conj())
        assert prod.abs() < 1e-4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()
    
    tester = TestMaths()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_orthgonalize_gs()
    tester.test_orthgonalize_svd()
