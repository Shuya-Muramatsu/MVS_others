import time
import json
import torch
import cv2
import numpy as np
from pathlib import Path


# 設定
INPUT_IMAGE = "../../NYUtest/nyu_sample/sample_00000.png"
OUTPUT_DEPTH = "depth.png"
TIMINGS_JSON = "timings.json"  # 計測結果を保存するファイル


def main(save_timings: bool = True):
    timings = {}
    start_total = time.time()

    # デバイス設定（GPU があれば GPU を使う）
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timings['device_selection'] = time.time() - t0

    # モデル読込（最軽量）: モデルを device に移して eval() にする
    t0 = time.time()
    midas = torch.hub.load("isl-org/MiDaS", "MiDaS_small").to(device).eval()
    transform = torch.hub.load("isl-org/MiDaS", "transforms").small_transform
    timings['model_load'] = time.time() - t0

    # 入力画像読み込み（BGR -> RGB）
    t0 = time.time()
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        raise FileNotFoundError(f"入力画像 {INPUT_IMAGE} が見つかりません")
    img = img[..., ::-1]
    timings['image_load'] = time.time() - t0

    # transform の出力次元に応じてバッチ次元を整える
    t0 = time.time()
    t = transform(img)
    if t.dim() == 3:
        inp = t.unsqueeze(0).to(device)
    elif t.dim() == 4:
        inp = t.to(device)
    else:
        raise RuntimeError(f"transform の返り値の次元が予期しない: {t.shape}")
    timings['transform'] = time.time() - t0

    # 推論
    t0 = time.time()
    with torch.no_grad():
        out = midas(inp)
    timings['inference'] = time.time() - t0

    # 出力整形
    t0 = time.time()
    depth = out.squeeze().cpu().numpy()
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
    cv2.imwrite(OUTPUT_DEPTH, (depth_vis * 255).astype(np.uint8))
    timings['postprocess_save'] = time.time() - t0

    timings['total'] = time.time() - start_total
    # モデルロード時間を除いた実測向けの合計
    timings['total_excl_model_load'] = timings['total'] - timings.get('model_load', 0.0)

    # 出力表示（見やすく、ミリ秒表記）
    def _ms(x):
        return x * 1000.0

    # JSON に ms 値を追加
    timings_ms = {f"{k}_ms": int(_ms(v)) for k, v in timings.items()}
    timings_out = {**timings, **timings_ms}

    # ANSI 色化（シンプル）
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(f"{CYAN}Timings:{RESET}")
    for k, v in timings.items():
        ms = int(_ms(v))
        print(f"  {GREEN}{k:25}{RESET} {YELLOW}{ms:8d} ms{RESET}  ({v:.4f} s)")

    if save_timings:
        try:
            with open(TIMINGS_JSON, 'w') as f:
                json.dump(timings_out, f, indent=2)
            print(f"Saved timings to {TIMINGS_JSON}")
        except Exception as e:
            print(f"Failed to save timings: {e}")
    


if __name__ == '__main__':
    main(save_timings=True)
