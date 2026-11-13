"""Dataset placement and distributed plumbing without full reconstructions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

import ptychi.api as api
import ptychi.parallel as parallel
from ptychi.device import AcceleratorModuleWrapper
from ptychi.io_handles import PtychographyDataset
from ptychi.reconstructors.ad_general import AutodiffReconstructor
from ptychi.reconstructors.lsqml import MultiprocessLSQMLReconstructor

pytestmark = pytest.mark.usefixtures("numerical_cpu")


@pytest.mark.parametrize("save_on_device", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dataset_placement_and_indexing(device, save_on_device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU storage")
    monkeypatch.setattr(AcceleratorModuleWrapper, "get_to_device_string", lambda: device)
    data = torch.arange(12.0).reshape(3, 2, 2)
    dataset = PtychographyDataset(data, fft_shift=False, save_data_on_device=save_on_device)
    assert dataset.patterns.device.type == (device if save_on_device else "cpu")
    indices, patterns = dataset[[2, 0]]
    torch.testing.assert_close(indices.cpu(), torch.tensor([2, 0]))
    torch.testing.assert_close(
        patterns.cpu(), torch.tensor([[[8.0, 9.0], [10.0, 11.0]], [[0.0, 1.0], [2.0, 3.0]]])
    )
    torch.testing.assert_close(data, torch.arange(12.0).reshape(3, 2, 2))


@pytest.mark.parametrize("rank,expected", [(0, [0, 1, 2]), (1, [3, 4])])
def test_distributed_batch_partition_handles_uneven_batches(rank, expected, monkeypatch):
    monkeypatch.setattr(parallel, "get_rank", lambda: rank)
    monkeypatch.setattr(parallel, "get_world_size", lambda: 2)
    chunk, sizes = parallel.MultiprocessMixin().get_chunk_of_current_rank(
        torch.arange(5), return_chunk_sizes=True
    )
    torch.testing.assert_close(chunk, torch.tensor(expected))
    assert sizes == [3, 2]


@pytest.mark.parametrize("batching", [api.BatchingModes.RANDOM, api.BatchingModes.COMPACT])
def test_distributed_lsqml_sums_object_and_averages_other_gradients(batching):
    recon = MultiprocessLSQMLReconstructor.__new__(MultiprocessLSQMLReconstructor)
    recon.current_epoch = 1
    recon.current_minibatch = 0
    recon.options = SimpleNamespace(
        batching_mode=batching, rescale_probe_intensity_in_first_epoch=False,
        exclude_measured_pixels_below=None,
    )
    recon.dataset = SimpleNamespace(valid_pixel_mask=None)
    recon.indices = []
    recon.dataloader = [None, None]
    recon.get_chunk_of_current_rank = lambda x, return_chunk_sizes=False: (
        (x[:2], [2, 1]) if return_chunk_sizes else x[:2]
    )
    recon.prepare_optimizer_closures = Mock()
    recon.forward_model = Mock(return_value=torch.tensor([[[1.0]], [[4.0]]]))
    recon.update_fourier_error = Mock()
    recon.compute_reconstruction_parameter_updates = Mock()
    recon.apply_reconstruction_parameter_updates = Mock()
    recon.reconstructor_buffers = Mock()
    recon.parameter_group = Mock()
    recon.loss_tracker = Mock()
    recon.run_minibatch([torch.tensor([0, 1, 2])], torch.tensor([[[1.0]], [[4.0]], [[9.0]]]))
    calls = recon.parameter_group.synchronize_optimizable_parameter_gradients.call_args_list
    assert calls[0].kwargs == {"op": dist.ReduceOp.AVG, "names_to_exclude": ["object"]}
    if batching == api.BatchingModes.RANDOM:
        assert calls[1].kwargs == {"op": dist.ReduceOp.SUM, "names_to_include": ["object"]}
    else:
        assert len(calls) == 1
    torch.testing.assert_close(recon.indices[0], torch.tensor([0, 1]))
    torch.testing.assert_close(
        recon.compute_reconstruction_parameter_updates.call_args.args[1],
        torch.tensor([[[1.0]], [[4.0]]]),
    )


def test_autodiff_builds_distributed_forward_model(monkeypatch):
    recon = AutodiffReconstructor.__new__(AutodiffReconstructor)
    model = torch.nn.Linear(2, 1)
    recon.forward_model_class = Mock(return_value=model)
    recon.forward_model_params = {"test_setting": 3}
    recon.parameter_group = object()
    wrapped = Mock()
    wrapper = Mock(return_value=wrapped)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", wrapper)
    recon.build_forward_model()
    recon.forward_model_class.assert_called_once_with(recon.parameter_group, test_setting=3)
    wrapper.assert_called_once_with(model)
    wrapped.to.assert_called_once_with(torch.device("cpu"))
    assert recon.forward_model is wrapped


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("mode", [api.BatchingModes.COMPACT, api.BatchingModes.UNIFORM])
def test_spatial_batch_samplers_cover_every_pattern_once(mode, batch_size):
    from ptychi.io_handles import PtychographyCompactBatchSampler, PtychographyUniformBatchSampler

    positions = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [10.0, 0.0], [10.0, 1.0]])
    sampler_type = (
        PtychographyCompactBatchSampler
        if mode == api.BatchingModes.COMPACT
        else PtychographyUniformBatchSampler
    )
    sampler = sampler_type(positions, batch_size=batch_size)
    batches = list(sampler)
    indices = torch.cat(batches)
    torch.testing.assert_close(indices.sort().values, torch.tensor([0, 1, 2, 3, 4]))
    assert len(batches) == len(sampler)
    assert all(0 < len(batch) <= batch_size for batch in batches)
