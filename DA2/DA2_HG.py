import time
import json
from pathlib import Path

from transformers import pipeline
from PIL import Image
import numpy as np


# 設定
INPUT_IMAGE = "../../NYUtest/nyu_sample/sample_00000.png"
OUTPUT_DEPTH = "depth_da2_small.png"
TIMINGS_JSON = "da2_timings.json"


def main(save_timings: bool = True):
    timings = {}
    start_total = time.time()

    # device 選択 (可能なら torch の cuda を使う)
    t0 = time.time()
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    timings["device_selection"] = time.time() - t0

    # モデルロード: pipeline の初期化を model_load とする
    t0 = time.time()
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    timings["model_load"] = time.time() - t0

    # 画像読み込み
    t0 = time.time()
    image = Image.open(INPUT_IMAGE).convert("RGB")
    timings["image_load"] = time.time() - t0

    # 前処理 (必要ならここで行うが、pipeline が内部で行うため計測のみ)
    t0 = time.time()
    # 例: numpy conversion を行わないが、計測枠として残す
    _ = np.array(image)
    timings["preprocess"] = time.time() - t0

    # 推論
    t0 = time.time()
    out = pipe(image)
    timings["inference"] = time.time() - t0

    # 後処理・保存
    t0 = time.time()
    # out["depth"] は PIL Image と仮定
    depth_pil = out.get("depth") if isinstance(out, dict) else out["depth"] if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "depth" in out[0] else None
    if depth_pil is None:
        # try different shapes
        try:
            depth_pil = out[0]["depth"]
        except Exception:
            raise RuntimeError("推論結果から depth を取得できませんでした: out type=" + str(type(out)))
    depth_pil.save(OUTPUT_DEPTH)
    timings["postprocess_save"] = time.time() - t0

    timings["total"] = time.time() - start_total
    timings["total_excl_model_load"] = timings["total"] - timings.get("model_load", 0.0)

    # 表示: ms 表記 + 色付き
    def _ms(x):
        return x * 1000.0

    timings_ms = {f"{k}_ms": int(_ms(v)) for k, v in timings.items()}
    timings_out = {**timings, **timings_ms}

    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(f"{CYAN}DA2 Timings:{RESET}")
    for k, v in timings.items():
        ms = int(_ms(v))
        print(f"  {GREEN}{k:25}{RESET} {YELLOW}{ms:8d} ms{RESET}  ({v:.4f} s)")

    if save_timings:
        try:
            with open(TIMINGS_JSON, "w") as f:
                json.dump(timings_out, f, indent=2)
            print(f"Saved timings to {TIMINGS_JSON}")
        except Exception as e:
            print(f"Failed to save timings: {e}")


if __name__ == '__main__':
    main(save_timings=True)
