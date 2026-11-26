"""학습된 교사 정책(ppo_teacher.py)을 카메라가 있는 환경에서 실행해 시연을 모은다.

실행 환경: conda activate maniskill
출력: replay_to_h5.py 와 같은 형식의 HDF5. 다음 단계(h5_to_lerobot.py)가 그대로 읽는다.

기록 방식
- 교사는 좌표를 보고 "지금 관절각에서 얼마나 움직일지"(증분)와 그리퍼를 열지 닫을지를 낸다. 학습 때와 똑같이
  관절 목표각(현재 관절각 + 증분, 절대값)과 그리퍼 명령(0 = 열림, 1 = 닫힘)으로 바꿔 pd_joint_pos 제어로 실행하고,
  그 값을 행동으로 기록한다. 자동 시연 데이터와 행동 정의가 같다.
- --noise 를 주면 일부 회차에서 팔을 가끔 "슬쩍 민다": 0.2~0.45초 동안 부드럽게 커졌다 줄어드는 치우침을 실행 행동에 더하고,
  기록은 치우침 없는 행동으로 한다(DART). 어긋난 상태에서 되돌아오는 행동이 데이터에 들어간다.
  (처음에는 매 스텝 독립인 잡음을 썼는데, 팔이 계속 떨리고 기록되는 목표각에도 떨림이 섞여 들어가서 바꿨다.)
- --max-toggles 를 주면 그리퍼 명령(열기/닫기)이 그보다 많이 바뀐 회차는 저장하지 않는다.
- 성공한 회차만 저장한다. 회차는 성공 판정이 난 스텝에서 끝난다.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils import common

from ppo_teacher import Agent
from replay_to_h5 import GRIPPER_CLOSED, project, to_np
from rl_place_env import RL_ENV_IDS

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"  # --video 로 mp4 를 저장할 때만 쓴다


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="ppo_teacher.py 가 저장한 latest.pt")
    p.add_argument("--object", required=True, help="cube, banana, mustard, baseball 중 하나")
    p.add_argument("--out", default="", help="출력 HDF5. 비우면 저장하지 않고 통계만 낸다")
    p.add_argument("--episodes", type=int, default=20, help="시도할 회차 수")
    p.add_argument("--seed-start", type=int, default=200000)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--wrist-camera", default="both", choices=["generic", "d435", "both"])
    p.add_argument("--noise", type=float, default=0.0, help="미는 세기: 치우침의 표준편차(행동 단위, 1 = 학습 때의 arm_step)")
    p.add_argument("--noise-frac", type=float, default=0.5, help="미는 회차의 비율")
    p.add_argument("--push-prob", type=float, default=0.03, help="미는 회차에서, 스텝마다 새로 밀기 시작할 확률")
    p.add_argument("--smooth", type=float, default=-1.0, help="팔 행동 저역 통과(0~1): 행동 = (1-s)*정책 출력 + s*직전 행동. 실행과 기록 모두 걸러진 행동을 쓴다. 음수면 교사가 학습 때 쓴 값")
    p.add_argument("--gripper-speed", type=float, default=-1.0, help="그리퍼가 닫히는 최대 속도(m/s). 0 이면 제한 없음. 음수면 교사가 학습 때 쓴 값")
    p.add_argument("--max-toggles", type=int, default=0, help="그리퍼 명령이 이보다 많이 바뀐 회차는 버린다 (0 = 거르지 않음)")
    p.add_argument("--visual-rand", type=float, default=0.0, help="화면 바꾸기의 세기(조명, 외부 카메라 위치·화각). 0 이면 끔")
    p.add_argument("--table-color", default=None, help="책상 색: 비우면 나무 무늬, white, white_mix(흰색 2/3 + 나무 1/6 + 다른 색 1/6)")
    p.add_argument("--video", default="", help="앞의 몇 회차를 mp4 로 저장할 경로(외부 + 손목 영상)")
    p.add_argument("--num-videos", type=int, default=6)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    env_id = RL_ENV_IDS[args.object]
    arm_step = ck["arm_step"]
    if args.smooth < 0:
        args.smooth = ck.get("smooth", 0.0)
    if args.gripper_speed < 0:
        args.gripper_speed = ck.get("gripper_speed", 0.0)
    onehot = torch.zeros(len(ck["objects"]))
    onehot[ck["objects"].index(args.object)] = 1.0
    agent = Agent(ck["obs_dim"], ck["act_dim"], ck["hidden"])
    agent.load_state_dict(ck["agent"])
    agent.eval()
    mean, std = ck["norm_mean"].cpu(), torch.sqrt(ck["norm_var"].cpu() + 1e-8)

    save = bool(args.out)
    render = save or bool(args.video)
    env = gym.make(env_id, obs_mode="rgb" if render else "state_dict", control_mode="pd_joint_pos", sim_backend="cpu",
                   wrist_camera=args.wrist_camera, reward_mode="sparse", gripper_speed=args.gripper_speed,
                   table_color=args.table_color, visual_rand=args.visual_rand,
                   **({"reconfiguration_freq": 1} if args.visual_rand > 0 or args.table_color == "white_mix" else {}))
    u = env.unwrapped
    rng = np.random.default_rng(args.seed_start)
    out = None
    if save:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = h5py.File(args.out, "w")
        out.attrs["fps"] = int(u.control_freq)
        out.attrs["env_id"] = env_id
        out.attrs["wrist_camera"] = args.wrist_camera
        out.attrs["source"] = f"teacher:{args.ckpt}"
    video = None

    kept, lengths, speeds, jerks, fails = 0, [], [], [], {"timeout": 0, "wrong_target": 0, "dropped": 0, "tipped": 0}
    toggles, chatter = [], 0
    for e in range(args.episodes):
        seed = args.seed_start + e
        obs, _ = env.reset(seed=seed)
        noisy = args.noise > 0 and rng.random() < args.noise_frac
        obs_params_base = obs["sensor_param"]["base_camera"]["extrinsic_cv"] if render else None  # 회차마다 외부 카메라 위치가 다를 수 있다
        if save and "base_camera_intrinsic_cv" not in out.attrs:
            for name, prm in obs["sensor_param"].items():
                if name not in ("hand_camera", "d435_camera"):
                    out.attrs[f"{name}_intrinsic_cv"] = to_np(prm["intrinsic_cv"])
                    out.attrs[f"{name}_extrinsic_cv"] = to_np(prm["extrinsic_cv"])
            out.attrs["robot_base_pos_world"] = to_np(u.agent.robot.pose.p)
        base_inv, base_pos = u.agent.robot.pose.inv(), to_np(u.agent.robot.pose.p)
        goal_world = to_np(u.goal_position())
        images, state, action, tcp, cube = {}, [], [], [], []
        success, dropped, tipped, info = False, False, False, u.get_info()
        prev_a = torch.zeros(ck["act_dim"])
        last_a = torch.zeros(ck["act_dim"])
        push, push_t, push_len = None, 0, 0
        for _ in range(args.max_steps):
            qpos = to_np(u.agent.robot.get_qpos())
            with torch.no_grad():
                o = common.flatten_state_dict(u._get_obs_state_dict(info), use_torch=True, device=u.device).cpu()[0]
                o = torch.cat([o, onehot, prev_a])
                a = agent.actor(torch.clamp((o - mean) / std, -10, 10)).clamp(-1, 1)
                if args.smooth > 0:
                    a[:6] = (1 - args.smooth) * a[:6] + args.smooth * last_a[:6]
                last_a = a.clone()
            run_a = a.clone()
            if noisy:  # 실행은 밀린 행동으로, 기록은 밀지 않은 행동으로
                if push is None and rng.random() < args.push_prob:
                    push, push_t, push_len = args.noise * torch.as_tensor(rng.standard_normal(6), dtype=torch.float32), 0, int(rng.integers(4, 10))
                if push is not None:
                    run_a[:6] = (a[:6] + push * float(np.sin(np.pi * (push_t + 1) / (push_len + 1)))).clamp(-1, 1)
                    push_t += 1
                    if push_t >= push_len:
                        push = None
            prev_a = run_a
            grip = GRIPPER_CLOSED if a[6] > 0 else 0.0
            clean = np.concatenate([qpos[:6] + arm_step * a[:6].numpy(), [grip]]).astype(np.float32)
            run = np.concatenate([qpos[:6] + arm_step * run_a[:6].numpy(), [grip]]).astype(np.float32)
            if render:
                for name, d in obs["sensor_data"].items():
                    images.setdefault(name.replace("_camera", ""), []).append(to_np(d["rgb"]))
            state.append(np.concatenate([qpos[:6], [qpos[6] / GRIPPER_CLOSED]]).astype(np.float32))
            action.append(np.concatenate([clean[:6], [clean[6] / GRIPPER_CLOSED]]).astype(np.float32))
            tcp.append(to_np(u.agent.tcp.pose.raw_pose).astype(np.float32))
            cube.append(to_np((base_inv * u.cube.pose).p).astype(np.float32))
            obs, _, _, _, info = env.step(run)
            if bool(info["success"]):
                success = True
                break
            if bool(info["fail"]):
                dropped, tipped = True, bool(info["tipped"])
                break
        if video is None and args.video and images:
            h, w = images["base"][0].shape[:2]
            video = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}",
                                      "-r", str(int(u.control_freq)), "-i", "-", "-pix_fmt", "yuv420p", args.video], stdin=subprocess.PIPE)
        if video is not None and e < args.num_videos:
            for b, hnd in zip(images["base"], images["hand"]):
                video.stdin.write(np.concatenate([b, hnd], 1).tobytes())
        if not success:
            fails["tipped" if tipped else "dropped" if dropped else "wrong_target" if bool(info.get("on_wrong_target", False)) else "timeout"] += 1
            continue
        act = np.stack(action)
        n_tog = int((np.diff((act[:, 6] > 0.5).astype(int)) != 0).sum())
        if args.max_toggles and n_tog > args.max_toggles:
            chatter += 1
            continue
        toggles.append(n_tog)
        lengths.append(len(act))
        d = np.diff(act[:, :6], axis=0)
        speeds.append(np.abs(d).max() * u.control_freq if len(d) else 0.0)
        jerks.append(np.abs(np.diff(d, axis=0)).mean() if len(d) > 1 else 0.0)
        if save:
            g = out.create_group(f"episode_{kept:06d}")
            g.attrs["seed"], g.attrs["task"], g.attrs["noisy"] = seed, u.task_text(), bool(noisy)
            g.attrs["table"] = getattr(u, "table_desc", "wood")
            g.attrs["base_camera_extrinsic_cv"] = to_np(obs_params_base) if obs_params_base is not None else np.zeros(0)
            if kept == 0:
                out.attrs["task"] = u.task_text()
            for k, v in images.items():
                g.create_dataset(k, data=np.stack(v), compression="gzip", compression_opts=1)
            g.create_dataset("state", data=np.stack(state))
            g.create_dataset("action", data=act)
            g.create_dataset("tcp_pose", data=np.stack(tcp))
            g.create_dataset("cube_pos", data=np.stack(cube))
            g.create_dataset("goal_pos", data=(goal_world - base_pos).astype(np.float32))
            K = to_np(obs["sensor_param"]["base_camera"]["intrinsic_cv"])
            E = to_np(obs["sensor_param"]["base_camera"]["extrinsic_cv"])
            g.create_dataset("goal_pixel", data=project(goal_world, E, K))
        kept += 1
    if video is not None:
        video.stdin.close()
        video.wait()
    if save:
        out.attrs["num_episodes"] = kept
        out.close()
    env.close()
    n = args.episodes
    print(f"{env_id} 잡음={args.noise}: 성공 {kept}/{n} ({kept / n:.1%}), 시간 초과 {fails['timeout']}, "
          f"다른 표시에 놓음 {fails['wrong_target']}, 떨어뜨림 {fails['dropped']}, 쓰러뜨림 {fails['tipped']}, 그리퍼 떨림으로 버림 {chatter}")
    if lengths:
        print(f"  길이 평균 {np.mean(lengths):.0f} / 최대 {np.max(lengths)} 스텝, 목표각 변화 최대 {np.max(speeds):.2f} rad/s, "
              f"목표각 2차 차분 평균 {np.mean(jerks) * 1000:.2f} mrad, 그리퍼 명령 바뀜 평균 {np.mean(toggles):.1f} / 최대 {np.max(toggles)} "
              f"(2회 이하 {np.mean(np.array(toggles) <= 2):.0%})" + (f" -> {args.out}" if save else ""))


if __name__ == "__main__":
    main()
