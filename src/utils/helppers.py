import numpy as np
import torch

def numpy2img_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    return img

def img_tensor2numpy(img: torch.Tensor) -> np.ndarray:
    img = img.cpu().detach().numpy()
    img = img.transpose((1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return img

def mask_patches(img, mask_ratio: float=0.75, patch_size: int=16):
    """
    画像 (B, C, H, W) をパッチに分割し、一定割合のパッチをマスクする関数。
    マスクは、元画像に対して masked 部分は0に置き換え、マスクテンソルも返す。
    マスクテンソルは、元々の画像と同じ空間サイズで、
    マスクされた箇所が0、非マスク箇所が1となる。
    """
    B, C, H, W = img.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    masks = []
    for _ in range(B):
        # 1次元のパッチマスク: 初めは全て1（残す）にして、ランダムに一部を0にする
        patch_mask = torch.ones(total_patches, device=img.device)
        num_masked = int(mask_ratio * total_patches)
        idx = torch.randperm(total_patches)[:num_masked]
        patch_mask[idx] = 0  # マスク対象は0

        # 2次元に変換し、各パッチを元のサイズに拡大する
        patch_mask = patch_mask.view(num_patches_h, num_patches_w)
        mask_img = patch_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
        masks.append(mask_img)
    masks = torch.stack(masks, dim=0)  # (B, H, W)
    masks = masks.unsqueeze(1)  # (B, 1, H, W)

    # マスクされた画像: マスク部分は0
    masked_img = img * masks
    return masked_img, masks
