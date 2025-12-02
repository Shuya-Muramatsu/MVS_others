import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2  # リポジトリ内のクラス

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) モデル設定
encoder = "vits"         # Small 用のエンコーダ（ViT-S）
model_type = "Small"     # or "Base", "Large"
model = DepthAnythingV2(encoder=encoder).to(device)
# チェックポイントパスは README / docs を参照して指定
ckpt_path = "checkpoints/depth_anything_v2_vits.pth"
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

# 2) 入力画像
img = cv2.imread("input.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3) 前処理（リポジトリのサンプルに合わせるのが安全）
img_input = cv2.resize(img_rgb, (518, 518))  # 例
img_input = img_input.astype(np.float32) / 255.0
img_input = (img_input - 0.5) / 0.5
img_input = np.transpose(img_input, (2, 0, 1))  # CHW
img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

# 4) 推論
with torch.no_grad():
    depth = model(img_input)["predicted_depth"]  # [1, 1, H', W'] など

# 5) 可視化
depth = torch.nn.functional.interpolate(
    depth,
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
)
depth = depth.squeeze().cpu().numpy()
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
depth_img = (depth_norm * 255).astype(np.uint8)
cv2.imwrite("depth_da2_small_official.png", depth_img)
