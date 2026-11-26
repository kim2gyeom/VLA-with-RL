"""자동 시연의 시뮬레이터 상태를 한 파일로 묶는다. 강화학습에서 "시연의 중간 상태에서 시작"할 때 쓴다.

실행 환경: conda activate maniskill
입력: gen_place_demos.py 가 만든 시연 원본(.h5 + .json). 매 스텝의 시뮬레이터 상태가 들어 있다.
출력: npz 한 개
  states/<종류>/<이름> : (전체 스텝 수, 차원)  모든 시연의 상태를 이어 붙인 것
  start, length        : 시연별 시작 위치와 길이
  goal_idx             : 시연별 목표 색(0 = 파랑, 1 = 노랑). 시연의 마지막 상태에서 물체가 놓인 표시로 정한다.
                         (시드로 환경을 다시 초기화하는 방법은 쓰지 않는다. 강화학습용 환경은 난수를 다르게 써서 색이 달라진다.)
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

TARGET_NAMES = ("target_blue", "target_yellow")


def kind_is_object(name):
    """물체(큐브 또는 YCB 물체)의 이름인지. 탁자와 표시는 아니다."""
    return name in ("cube", "object")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--env-id", default="", help="쓰지 않는다(예전 명령과의 호환용)")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    raw = h5py.File(args.raw, "r")
    episodes = json.loads(Path(args.raw).with_suffix(".json").read_text())["episodes"]
    if args.limit:
        episodes = episodes[: args.limit]

    chunks, start, length, goal, miss = {}, [], [], [], []
    offset = 0
    for ep in episodes:
        g = raw[f"traj_{ep['episode_id']}"]["env_states"]
        n, final = None, {}
        for kind in ("actors", "articulations"):
            for name, arr in g[kind].items():
                a = arr[:]
                n = len(a)
                final[name] = a[-1, :2]
                chunks.setdefault(f"{kind}/{name}", []).append(a.astype(np.float32))
        obj = next(v for k, v in final.items() if kind_is_object(k))
        dist = [float(np.linalg.norm(obj - final[t])) for t in TARGET_NAMES]
        goal.append(int(np.argmin(dist)))
        miss.append(min(dist))
        start.append(offset)
        length.append(n)
        offset += n
    out = {f"states/{k}": np.concatenate(v) for k, v in chunks.items()}
    out.update(start=np.array(start), length=np.array(length), goal_idx=np.array(goal))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"시연 {len(start)}개, 상태 {offset}개, 길이 평균 {np.mean(length):.0f} / 최대 {np.max(length)}, "
          f"목표가 노랑인 비율 {np.mean(goal):.2f}, 마지막 상태에서 물체와 목표 표시의 거리 평균 {np.mean(miss) * 1000:.1f} mm "
          f"/ 최대 {np.max(miss) * 1000:.1f} mm -> {args.out}")


if __name__ == "__main__":
    main()
