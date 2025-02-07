import argparse

import torch

import ptychi.image_proc as ip
import test_utils as tutils


class TestPatchExtractionPlacement(tutils.BaseTester):
    
    def test_patch_extraction_placement_set_ordinary(self):
        torch.manual_seed(123)
        
        positions = torch.rand(10, 2) * 64 + 32
        image = torch.rand(128, 128)
        
        patches = ip.extract_patches_fourier_shift(image, positions, (32, 32))
        image_replaced = ip.place_patches_fourier_shift(torch.zeros_like(image), positions, patches, op="set", adjoint_mode=False)
        
        mask = image_replaced > 0
        orig_sum = torch.sum(image[mask])
        replaced_sum = torch.sum(image_replaced[mask])
        
        print("Mode: set")
        print("Sum of original image ROI:", orig_sum)
        print("Sum of replaced image ROI:", replaced_sum)
        print("Max of absolute difference:", torch.max(torch.abs(image_replaced[mask] - image[mask])))
        
        if self.debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image)
            ax[1].imshow(image_replaced)
            plt.suptitle("Mode: set")
            plt.tight_layout()
            plt.show()
        
        assert torch.allclose(orig_sum, replaced_sum, rtol=1e-2)
        
    def test_patch_extraction_placement_add_adjoint(self):
        torch.manual_seed(123)
        
        positions = torch.tensor([[64, 64]])
        image = torch.zeros([128, 128])
        nonzero_rad = 15
        image[64 - nonzero_rad:64 + nonzero_rad, 64 - nonzero_rad:64 + nonzero_rad] = torch.rand(2 * nonzero_rad, 2 * nonzero_rad)
        
        patches = ip.extract_patches_fourier_shift(image, positions, (32, 32))
        image_replaced = ip.place_patches_fourier_shift(torch.zeros_like(image), positions, patches, op="add", adjoint_mode=True)
                
        if self.debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image)
            ax[1].imshow(image_replaced)
            plt.suptitle("Mode: add")
            plt.tight_layout()
            plt.show()
        
        # Verify if extraction and placement are Hermitian adjoints using the identity 
        #   <Ax, y> = <x, A^H y>
        # where A is the extraction operator, A^H is the placement operator, 
        # x is the original image, and y is the extracted patch; y = Ax.
        ax_y = torch.sum(patches.abs() ** 2)
        x_ahy = torch.sum(image_replaced * image)
        assert torch.allclose(ax_y, x_ahy)
        
    def test_patch_extraction_placement_int(self):        
        positions = torch.tensor([[5, 5], [20, 20], [15, 20]]) + 0.5
        image = torch.arange(32 * 32).reshape(32, 32)
        
        patches = ip.extract_patches_integer(image, positions, (8, 8))
        image_replaced = ip.place_patches_integer(torch.zeros_like(image), positions, patches, op="add")
                
        ax_y = torch.sum(patches.abs() ** 2)
        x_ahy = torch.sum(image_replaced * image)
        assert torch.allclose(ax_y, x_ahy)
        assert torch.all(patches.sum(-1).sum(-1) == torch.tensor([11616, 43296, 33056]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()
    
    tester = TestPatchExtractionPlacement()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_patch_extraction_placement_set_ordinary()
    tester.test_patch_extraction_placement_add_adjoint()
    tester.test_patch_extraction_placement_int()
