import argparse

import torch

import ptychi.image_proc as ip
import test_utils as tutils


class TestRemovePolynomialBackground(tutils.BaseTester):
    def test_remove_polynomial_background_linear_32x32(self):
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        img = 2.0 * y + 3.0 * x + 5.0
        img = img.to(torch.float32)
        flat_region_mask = torch.ones_like(img, dtype=torch.bool)

        result = ip.remove_polynomial_background(
            img=img, flat_region_mask=flat_region_mask, polyfit_order=1
        )

        if self.debug:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img.detach().cpu().numpy())
            ax[0].set_title("Input")
            ax[1].imshow(result.detach().cpu().numpy())
            ax[1].set_title("Output")
            plt.tight_layout()
            plt.show()

        assert torch.max(torch.abs(result)) < 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    args = parser.parse_args()

    tester = TestRemovePolynomialBackground()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_remove_polynomial_background_linear_32x32()
