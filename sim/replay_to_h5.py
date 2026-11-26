"""자동 생성한 시연(행동만 저장됨)을 두 카메라 환경에서 다시 실행해 영상·상태·행동을 모은다.

실행 환경: conda activate maniskill
출력: 에피소드별 그룹을 가진 HDF5 한 개. 다음 단계(h5_to_lerobot.py)가 이것을 읽는다.

저장 정의
- 영상: 환경의 모든 카메라(v1: base, hand / v2: base(비스듬한 시점), hand, top)
- goal_pos (3): 목표 지점의 좌표(m, 로봇 베이스 기준). goal_pixel (2): 외부(base) 영상에서 목표의 픽셀 위치
- cube_pos (3): 큐브 좌표(m, 로봇 베이스 기준). 고정 카메라의 내부·외부 파라미터는 파일 속성에 기록
- state  (7): 팔 관절각 6개(rad) + 그리퍼 열림 정도(0=열림, 1=완전히 닫힘)
- action (7): 팔 관절 목표각 6개(rad, 절대값) + 그리퍼 명령(0=열림, 1=닫힘)
- 제어 주기 20 Hz, 영상 256x256 RGB 두 개(base, hand)
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from tqdm import tqdm

import place_cube_env  # noqa: F401  (환경 등록)
import place_object_env  # noqa: F401
from pickcube_xarm6_env import ENV_ID

GRIPPER_CLOSED = 0.81  # xarm6_robotiq 그리퍼 관절의 최대 명령값(rad)


def to_np(x):
    return x[0].cpu().numpy()


def read_obs(obs):
    qpos = to_np(obs["agent"]["qpos"])
    state = np.concatenate([qpos[:6], [qpos[6] / GRIPPER_CLOSED]]).astype(np.float32)
    images = {name.replace("_camera", ""): to_np(d["rgb"]) for name, d in obs["sensor_data"].items()}
    return images, state, to_np(obs["extra"]["tcp_pose"]).astype(np.float32)


def project(point_world, extrinsic_cv, intrinsic_cv):
    """세계 좌표의 점을 카메라 영상의 픽셀 (u, v) 로 투영한다."""
    cam = extrinsic_cv @ np.append(point_world, 1.0)
    uv = intrinsic_cv @ cam
    return (uv[:2] / uv[2]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="motionplanning 이 만든 .h5 경로 (.json 이 옆에 있어야 함)")
    p.add_argument("--out", required=True, help="출력 HDF5 경로")
    p.add_argument("--limit", type=int, default=0, help="앞에서부터 이 개수만 처리 (0 = 전부)")
    p.add_argument("--start", type=int, default=0, help="이 번호의 시연부터 처리(여러 프로세스로 나눠 돌릴 때)")
    p.add_argument("--wrist-camera", default="generic", choices=["generic", "d435", "both"],
                   help="손목 카메라. d435 는 UFACTORY 순정 거치대에 단 RealSense D435, both 는 둘 다 기록(hand, d435)")
    p.add_argument("--env-id", default=ENV_ID, help="영상을 기록할 환경. v2 는 비스듬한 외부 시점")
    args = p.parse_args()

    raw = h5py.File(args.raw, "r")
    meta = json.loads(Path(args.raw).with_suffix(".json").read_text())
    episodes, seen = [], set()
    for ep in meta["episodes"]:  # 병렬 생성 때 겹친 시드는 한 번만 쓴다
        if ep["episode_seed"] not in seen:
            seen.add(ep["episode_seed"])
            episodes.append(ep)
    episodes = episodes[args.start:]
    if args.limit:
        episodes = episodes[: args.limit]

    env = gym.make(args.env_id, obs_mode="rgb", control_mode="pd_joint_pos", sim_backend="cpu", wrist_camera=args.wrist_camera)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = h5py.File(args.out, "w")
    out.attrs["fps"] = int(env.unwrapped.control_freq)
    out.attrs["env_id"] = args.env_id
    out.attrs["wrist_camera"] = args.wrist_camera

    u = env.unwrapped
    kept = dropped = 0
    for ep in tqdm(episodes):
        actions = raw[f"traj_{ep['episode_id']}"]["actions"][:]
        obs, _ = env.reset(seed=ep["episode_seed"])
        if kept == 0 and dropped == 0:  # 고정 카메라의 내부·외부 파라미터를 한 번 기록한다
            for name, prm in obs["sensor_param"].items():
                if name not in ("hand_camera", "d435_camera"):
                    out.attrs[f"{name}_intrinsic_cv"] = to_np(prm["intrinsic_cv"])
                    out.attrs[f"{name}_extrinsic_cv"] = to_np(prm["extrinsic_cv"])  # 세계 -> 카메라
            out.attrs["robot_base_pos_world"] = to_np(u.agent.robot.pose.p)
        K = to_np(obs["sensor_param"]["base_camera"]["intrinsic_cv"])
        E = to_np(obs["sensor_param"]["base_camera"]["extrinsic_cv"])
        base_inv = u.agent.robot.pose.inv()
        goal_world = to_np(u.goal_position())
        base_pos = to_np(u.agent.robot.pose.p)  # 로봇 베이스는 회전 없이 놓여 있어 위치만 빼면 된다
        goal_base = (goal_world - base_pos).astype(np.float32)  # 로봇 베이스 기준 (m)
        task = u.task_text()
        goal_pixel = project(goal_world, E, K)  # 외부(base) 영상에서의 픽셀
        images, state, tcp, cube = {}, [], [], []
        info = {}
        for a in actions:  # t 시점 관측과 t 시점 행동을 짝짓는다
            im, s_, t_ = read_obs(obs)
            for k, v in im.items():
                images.setdefault(k, []).append(v)
            state.append(s_); tcp.append(t_)
            cube.append(to_np((base_inv * u.cube.pose).p).astype(np.float32))
            obs, _, _, _, info = env.step(a)
        if not bool(info["success"]):  # 재실행에서 성공이 재현되지 않으면 버린다
            dropped += 1
            continue
        act = actions.astype(np.float32).copy()
        act[:, 6] /= GRIPPER_CLOSED
        g = out.create_group(f"episode_{kept:06d}")
        g.attrs["seed"] = ep["episode_seed"]
        g.attrs["task"] = task  # 지시문은 회차마다 다를 수 있다
        if kept == 0:
            out.attrs["task"] = task
        for k, v in images.items():
            g.create_dataset(k, data=np.stack(v), compression="gzip", compression_opts=1)
        g.create_dataset("state", data=np.stack(state))
        g.create_dataset("action", data=act)
        g.create_dataset("tcp_pose", data=np.stack(tcp))
        g.create_dataset("cube_pos", data=np.stack(cube))   # 로봇 베이스 기준 (m), 프레임별
        g.create_dataset("goal_pos", data=goal_base)         # 로봇 베이스 기준 (m), 에피소드 내 고정
        g.create_dataset("goal_pixel", data=goal_pixel)      # base 영상의 (u, v), 에피소드 내 고정
        kept += 1
    out.attrs["num_episodes"] = kept
    out.close()
    env.close()
    print(f"저장 {kept}개, 재현 실패로 제외 {dropped}개 -> {args.out}")


if __name__ == "__main__":
    main()
