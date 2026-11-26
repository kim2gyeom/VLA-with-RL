"""물체별·조각별로 변환한 LeRobot 데이터셋을 하나로 합치고, 점검용으로 빼 둘 에피소드 목록을 적는다.

합치는 순서는 조각 0 의 네 물체, 조각 1 의 네 물체, … 이다. 마지막 조각에서 물체마다 끝의 12개씩을
meta/holdout.json 에 적어 둔다(학습 스크립트가 이 목록을 학습에서 뺀다).
"""
import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OBJECTS = ["cube", "banana", "mustard", "baseball"]
HOLDOUT_PER_OBJECT = 12

p = argparse.ArgumentParser()
p.add_argument("--work", required=True)
p.add_argument("--name", required=True)
p.add_argument("--shards", type=int, required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
parts, holdout, offset, counts = [], [], 0, {}
for i in range(args.shards):
    for k in OBJECTS:
        ds = LeRobotDataset(f"local/{args.name}__{k}.{i}", root=Path(args.work) / f"{args.name}__{k}.{i}")
        n = ds.meta.total_episodes
        counts[k] = counts.get(k, 0) + n
        if i == args.shards - 1:
            holdout += list(range(offset + n - HOLDOUT_PER_OBJECT, offset + n))
        offset += n
        parts.append(ds)
merged = merge_datasets(parts, output_repo_id=f"local/{args.name}_sim", output_dir=out)
(out / "meta" / "holdout.json").write_text(json.dumps(holdout))
cam = Path(args.work) / f"{args.name}__cube.0" / "meta" / "cameras.json"
if cam.exists():
    shutil.copy(cam, out / "meta" / "cameras.json")
print(f"합친 결과: 에피소드 {merged.meta.total_episodes}개 {counts}, 프레임 {merged.meta.total_frames}개, "
      f"지시문 {len(merged.meta.tasks)}종, 점검용 {len(holdout)}개 -> {out}")
