"""
Thin wrapper around AuxOL's online-learning core.

AuxOL (github.com/qianxihaoyue/AuxOL, IEEE-TMI 2025) pairs SAM with a small
auxiliary UNet trained online, one human-corrected crop at a time. Each
correction does two things:
  1. Pushes (crop, corrected mask) onto a replay buffer and takes one SGD
     step of the UNet over that buffer (structure_loss, as in the paper).
  2. Grid-searches, in hindsight, the SAM/UNet fusion weight (alpha) that
     would have maximized Dice against *this* correction.

The alpha actually applied to a given sample is the running average of
*past* corrections' best alphas — never the current one — so scoring an
as-yet-uncorrected sample never leaks its own ground truth. This mirrors
the "adaptive segmentation fusion" described on the AuxOL project page.

The repo has no LICENSE file (GitHub reports license: None), so its code
is not vendored here — it's added as a git submodule at
third_party/AuxOL and imported at runtime, same as any external
dependency.

Unit of work: this wrapper operates on whatever crop run_sam.py already
produces (the *padded* detection cutout, not a tighter bbox-only crop like
AuxOL's own dataset loader uses) — image, SAM logits, and corrected mask
must all share that cutout's H×W.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

_AUXOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "AuxOL")


def _load_auxol():
    if _AUXOL_DIR not in sys.path:
        sys.path.insert(0, _AUXOL_DIR)
    if not os.path.isdir(_AUXOL_DIR):
        raise RuntimeError(
            f"{_AUXOL_DIR} not found — run "
            "`git submodule update --init third_party/AuxOL` first."
        )
    from Nets.Unet.unet_model import UNet
    from tool import structure_loss, binary
    return UNet, structure_loss, binary


def calc_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Standard Dice/F1 overlap; both arrays assumed in [0, 1]."""
    pred = pred.reshape(1, -1)
    gt = gt.reshape(1, -1)
    smooth = 1.0
    intersection = (pred * gt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


class AuxOLOnlineUpdater:
    """
    Streams one detection crop at a time through AuxOL's online-learning
    loop. State (UNet weights, optimizer, replay buffer, alpha history)
    persists across calls, and across save_state()/load_state(), so it
    should keep improving the more corrections you feed it — within a
    session and across sessions if you reload the checkpoint.

    Feed samples through .step() in the order you corrected them —
    AuxOL is an online method, its history is order-dependent.
    """

    def __init__(self, four_channel: bool = True, crop_size: int = 128,
                 lr: float = 5e-4, replay_size: int = 32, alpha_num: int = 5,
                 alpha_percent: float = 1.0, device: str = None):
        UNet, structure_loss, binary = _load_auxol()
        self._structure_loss = structure_loss
        self._binary = binary

        self.four_channel = four_channel
        self.crop_size = crop_size
        self.replay_size = replay_size
        self.alpha_num = alpha_num
        self.alpha_percent = alpha_percent
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        in_channels = 4 if four_channel else 3
        self.net = UNet(n_channels=in_channels, n_classes=1, bilinear=False).to(self.device)
        self.net.train()
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.0005)

        self._batch_images = []   # replay buffer, each [1,C,crop,crop]
        self._batch_masks = []    # replay buffer, each [1,1,crop,crop]
        self._alphas = []         # per-sample best alpha, in hindsight (uses that sample's GT)
        self._mean_alphas = []    # per-sample alpha actually applied (no current-GT leakage)

        self._resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])

    # -- internal --------------------------------------------------------

    def _prep_image(self, crop_rgb_u8: np.ndarray, sam_binary_u8: np.ndarray) -> torch.Tensor:
        img = crop_rgb_u8
        if self.four_channel:
            fourth = (sam_binary_u8.astype(np.uint8) * 255)[:, :, None]
            img = np.concatenate([img, fourth], axis=2)
        return self._resize(img).unsqueeze(0).to(self.device)

    def _prep_mask(self, mask01: np.ndarray) -> torch.Tensor:
        m = cv2.resize(mask01.astype(np.float32), (self.crop_size, self.crop_size))
        m = (m >= 0.5).astype(np.float32)
        return torch.from_numpy(m)[None, None, :, :].to(self.device)

    def _unet_forward_train(self, image_t: torch.Tensor, mask_t: torch.Tensor, h: int, w: int):
        """Push (image_t, mask_t) onto the replay buffer, take one SGD step
        over the whole buffer, and return this sample's UNet logits at its
        native (h, w) resolution plus the training loss."""
        if len(self._batch_images) >= self.replay_size:
            self._batch_images.pop(0)
            self._batch_masks.pop(0)
        self._batch_images.append(image_t)
        self._batch_masks.append(mask_t)

        images = torch.cat(self._batch_images, dim=0)
        masks = torch.cat(self._batch_masks, dim=0)

        self.optimizer.zero_grad()
        output = self.net(images)
        loss = self._structure_loss(output, masks)
        loss.backward()
        self.optimizer.step()

        last = output[-1:].detach()
        last = F.interpolate(last, size=(h, w), mode="bilinear")
        return last.cpu().squeeze(0).squeeze(0).numpy(), float(loss.item())

    def _unet_forward_infer(self, image_t: torch.Tensor, h: int, w: int) -> np.ndarray:
        with torch.no_grad():
            output = self.net(image_t)
            output = F.interpolate(output, size=(h, w), mode="bilinear")
        return output.cpu().squeeze(0).squeeze(0).numpy()

    def _pick_alpha(self, sam_logits: np.ndarray, unet_logits: np.ndarray, gt01: np.ndarray) -> float:
        best_alpha, best_dice = 1.0, -1.0
        for a in [round(x, 2) for x in np.arange(0, self.alpha_percent + 0.01, 0.05).tolist()]:
            fused = torch.from_numpy(a * sam_logits + (1 - a) * unet_logits).sigmoid().numpy()
            d = calc_dice(self._binary(fused), gt01)
            if d > best_dice:
                best_alpha, best_dice = a, d
        return best_alpha

    def _current_alpha(self) -> float:
        n = len(self._alphas)
        if n <= self.alpha_num:
            return 1.0   # cold start: pure SAM, no fusion yet
        return float(np.mean(self._alphas[-1 - self.alpha_num:-1]))

    def _fuse(self, sam_logits: np.ndarray, unet_logits: np.ndarray, alpha: float):
        prob = torch.from_numpy(alpha * sam_logits + (1 - alpha) * unet_logits).sigmoid().numpy()
        return prob, self._binary(prob)

    # -- public API --------------------------------------------------------

    def step(self, crop_rgb_u8: np.ndarray, sam_logits: np.ndarray,
              corrected_mask01: np.ndarray) -> dict:
        """
        Feed one human-corrected detection crop into the online learner.

        crop_rgb_u8:      HxWx3 uint8 cutout image.
        sam_logits:       HxW float32, SAM's raw pre-sigmoid output for
                           this same crop (as saved by run_sam.py's
                           <tag>_logits.fits).
        corrected_mask01: HxW {0,1}, the human-rectified ground truth.

        Returns fused mask/probability at native resolution, the alpha
        applied (from past corrections only) and the hindsight-best alpha
        for this one, the replay loss, and Dice(SAM vs correction) /
        Dice(fused vs correction) for logging.
        """
        h, w = corrected_mask01.shape
        sam_bin = (sam_logits > 0).astype(np.uint8)

        image_t = self._prep_image(crop_rgb_u8, sam_bin)
        mask_t = self._prep_mask(corrected_mask01)
        unet_logits, loss = self._unet_forward_train(image_t, mask_t, h, w)

        alpha_applied = self._current_alpha()
        fused_prob, fused_mask = self._fuse(sam_logits, unet_logits, alpha_applied)

        hindsight_alpha = self._pick_alpha(sam_logits, unet_logits, corrected_mask01)
        self._alphas.append(hindsight_alpha)
        self._mean_alphas.append(alpha_applied)

        return {
            "fused_mask": fused_mask.astype(np.uint8),
            "fused_prob": fused_prob.astype(np.float32),
            "alpha_applied": alpha_applied,
            "alpha_hindsight": hindsight_alpha,
            "replay_loss": loss,
            "dice_sam_vs_correction": calc_dice(sam_bin.astype(np.float32), corrected_mask01.astype(np.float32)),
            "dice_fused_vs_correction": calc_dice(fused_mask.astype(np.float32), corrected_mask01.astype(np.float32)),
        }

    def predict_only(self, crop_rgb_u8: np.ndarray, sam_logits: np.ndarray) -> dict:
        """
        Apply the current adapted model to a crop with no correction yet
        (e.g. to preview AuxOL's improved mask before deciding whether to
        correct it). Does not update any state.
        """
        h, w = sam_logits.shape
        sam_bin = (sam_logits > 0).astype(np.uint8)
        image_t = self._prep_image(crop_rgb_u8, sam_bin)
        unet_logits = self._unet_forward_infer(image_t, h, w)

        alpha_applied = self._current_alpha()
        fused_prob, fused_mask = self._fuse(sam_logits, unet_logits, alpha_applied)
        return {
            "fused_mask": fused_mask.astype(np.uint8),
            "fused_prob": fused_prob.astype(np.float32),
            "alpha_applied": alpha_applied,
        }

    # -- persistence --------------------------------------------------------

    def save_state(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "alphas": self._alphas,
            "mean_alphas": self._mean_alphas,
            "batch_images": [t.cpu() for t in self._batch_images],
            "batch_masks": [t.cpu() for t in self._batch_masks],
            "four_channel": self.four_channel,
            "crop_size": self.crop_size,
        }, path)

    def load_state(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._alphas = ckpt["alphas"]
        self._mean_alphas = ckpt["mean_alphas"]
        self._batch_images = [t.to(self.device) for t in ckpt["batch_images"]]
        self._batch_masks = [t.to(self.device) for t in ckpt["batch_masks"]]
