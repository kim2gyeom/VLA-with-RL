"""replay_to_h5.py 의 출력을 LeRobot v3.0 데이터셋으로 바꾼다.

실행 환경: conda activate xarm6-pi05
결과 폴더는 lerobot-train 의 --dataset.root 로 바로 쓸 수 있다.
"""
import argparse
import shutil
from pathlib import Path

import h5py
import json
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

NAMES = ["J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "gripper.pos"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--root", required=True, help="만들 데이터셋 폴더. 이미 있으면 --overwrite 가 필요하다.")
    p.add_argument("--repo-id", default="local/pickcube_xarm6_sim")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="이 번호의 에피소드부터 변환(여러 프로세스로 나눠 돌릴 때)")
    p.add_argument("--fast", action="store_true", help="영상을 받는 즉시 인코딩한다(중간 PNG 를 쓰지 않아 빠르다)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cameras", default="base,hand", help="모델 입력으로 넣을 카메라. 예: base,hand,top. 'hand:d435' 처럼 쓰면 기록 파일의 d435 영상을 hand 라는 이름으로 넣는다")
    p.add_argument("--state-with-goal", action="store_true",
                   help="상태 뒤에 목표 좌표 3개를 붙인다(ManiSkill 공식 기준처럼 목표를 숫자로 주는 방식)")
    args = p.parse_args()

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} 가 이미 있습니다. 다시 만들려면 --overwrite 를 붙이세요.")
        shutil.rmtree(root)

    src = h5py.File(args.h5, "r")
    file_task, fps = src.attrs["task"], int(src.attrs["fps"])
    keys = sorted(src.keys())[args.start:]
    if args.limit:
        keys = keys[: args.limit]
    def video(cam):  # 카메라마다 영상 크기가 다를 수 있다 (예: D435 손목 카메라는 320x240)
        h, w = src[keys[0]][cam].shape[1:3]
        return {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}

    cams = {}  # 데이터셋에서의 이름 -> 기록 파일에서의 이름
    for c in args.cameras.split(","):
        name, _, source = c.strip().partition(":")
        cams[name] = source or name
    has_goal = "goal_pos" in src[keys[0]]
    state_names = NAMES + (["goal.x", "goal.y", "goal.z"] if args.state_with_goal else [])
    features = {f"observation.images.{c}": video(src_name) for c, src_name in cams.items()}
    features["observation.state"] = {"dtype": "float32", "shape": (len(state_names),), "names": state_names}
    features["action"] = {"dtype": "float32", "shape": (7,), "names": NAMES}
    if has_goal:  # 모델 입력이 아닌 참고용 열. 이름이 observation/action 으로 시작하지 않아 학습기가 무시한다.
        features["goal.pos"] = {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]}
        features["goal.pixel"] = {"dtype": "float32", "shape": (2,), "names": ["u", "v"]}
    ds = LeRobotDataset.create(args.repo_id, fps, features=features, root=root,
                               robot_type="xarm6_robotiq_sim", use_videos=True,
                               streaming_encoding=args.fast, encoder_threads=2 if args.fast else None)
    try:
        for k in tqdm(keys):
            g = src[k]
            task = g.attrs.get("task", file_task)
            imgs = {c: g[src_name][:] for c, src_name in cams.items()}
            state, action = g["state"][:], g["action"][:]
            goal = g["goal_pos"][:] if has_goal else None
            if args.state_with_goal:
                state = np.concatenate([state, np.repeat(goal[None], len(state), 0)], 1).astype(np.float32)
            for i in range(len(action)):
                frame = {f"observation.images.{c}": imgs[c][i] for c in cams}
                frame.update({"observation.state": state[i], "action": action[i], "task": task})
                if has_goal:
                    frame.update({"goal.pos": goal, "goal.pixel": g["goal_pixel"][:]})
                ds.add_frame(frame)
            ds.save_episode()
    finally:
        ds.finalize()
    cam = {k: np.asarray(v).tolist() for k, v in src.attrs.items() if "trinsic" in k or k == "robot_base_pos_world"}
    if cam:  # 좌표와 픽셀을 서로 바꿀 때 쓰는 카메라 파라미터
        (root / "meta" / "cameras.json").write_text(json.dumps(cam, indent=2))
    print(f"에피소드 {ds.num_episodes}개, 프레임 {ds.num_frames}개, {fps} Hz -> {root}")


if __name__ == "__main__":
    main()
