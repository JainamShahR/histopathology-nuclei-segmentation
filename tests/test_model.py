import torch
from histoseg.model import UNet
from histoseg.metrics import batch_dice_iou, segmentation_loss


def test_unet_returns_correct_shape_even_for_odd_dimensions():
    net = UNet(base_channels=4)
    with torch.no_grad():
        result = net(torch.randn(2, 3, 65, 67))
    assert result.shape == (2, 1, 65, 67)


def test_metrics_for_perfect_predictions_and_empty_masks():
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]], [[[-10.0, -10.0], [-10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
    dice, iou = batch_dice_iou(logits, targets)
    assert torch.allclose(dice, torch.ones(2))
    assert torch.allclose(iou, torch.ones(2))
    assert torch.isfinite(segmentation_loss(logits, targets))


def test_metrics_mismatch_known_result():
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    dice, iou = batch_dice_iou(logits, targets)
    assert abs(dice.item() - 0.5) < 1e-6
    assert abs(iou.item() - 1.0 / 3.0) < 1e-6
