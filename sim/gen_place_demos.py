"""집어서 놓기 과제(PlaceCubeXArm6VLA-v1)의 시연을 자동 동작으로 만든다. 성공한 시연만 저장한다.

실행 환경: conda activate maniskill
출력: <record-dir>/<name>.h5 와 .json (시작 시드와 로봇 명령). 영상은 replay_to_h5.py 로 따로 기록한다.
"""
import argparse
import multiprocessing as mp
import os
import os.path as osp

import gymnasium as gym
import numpy as np
import sapien
from tqdm import tqdm

from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.xarm6.motionplanner import XArm6RobotiqMotionPlanningSolver
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.utils.wrappers.record import RecordEpisode

import place_object_env  # noqa: F401  (물체별 환경 등록)
from place_cube_env import PLACE_ENV_ID


def solve(env, seed):
    """접근 → 잡기 → 들기 → 표시 위로 이동 → 내려놓기 → 그리퍼 열기 → 손 빼기."""
    env.reset(seed=seed)
    u = env.unwrapped
    planner = XArm6RobotiqMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False)

    def move(pose):  # 직선에 가까운 경로를 먼저 쓰고, 안 되면 탐색 경로로 대신한다
        res = planner.move_to_pose_with_screw(pose)
        return planner.move_to_pose_with_RRTStar(pose) if res == -1 else res

    try:
        approaching = np.array([0, 0, -1])
        if hasattr(u, "grasp_spec"):  # 생활용품: 물체마다 정해 둔 위치를 위에서 집는다
            center, closing = u.grasp_spec()
            grasp_pose = u.agent.build_grasp_pose(approaching, closing, center)
            top = u.object_top_z()
            if top > 0.10:  # 높은 물체는 먼저 곧장 위로 올라간 뒤 접근해, 가는 길에 물체를 치지 않게 한다
                tcp = u.agent.tcp.pose.sp
                if move(sapien.Pose([tcp.p[0], tcp.p[1], top + 0.12], tcp.q)) == -1:
                    return -1
        else:  # 큐브: 감싸는 상자를 기준으로 잡는다
            target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
            grasp = compute_grasp_info_by_obb(get_actor_obb(u.cube), approaching=approaching,
                                              target_closing=target_closing, depth=0.025)
            grasp_pose = u.agent.build_grasp_pose(approaching, grasp["closing"], u.cube.pose.sp.p)

        if move(grasp_pose * sapien.Pose([0, 0, -0.05])) == -1:
            return -1
        if move(grasp_pose) == -1:
            return -1
        planner.close_gripper(t=10)
        lift_pose = sapien.Pose([0, 0, 0.10]) * grasp_pose
        if move(lift_pose) == -1:
            return -1

        goal = u.goal_position()[0].cpu().numpy()
        cube = u.cube.pose.sp.p
        offset = np.array([goal[0] - cube[0], goal[1] - cube[1], 0.0])
        if move(sapien.Pose(lift_pose.p + offset, lift_pose.q)) == -1:
            return -1
        place_pose = sapien.Pose(grasp_pose.p + offset + np.array([0, 0, 0.004]), grasp_pose.q)
        if move(place_pose) == -1:
            return -1
        planner.open_gripper()
        return move(sapien.Pose([0, 0, 0.08]) * place_pose)
    finally:
        planner.close()


def _main(args, proc_id=0):
    name = args.traj_name + (f".{proc_id}" if args.num_procs > 1 else "")
    env = gym.make(args.env_id, obs_mode="none", control_mode="pd_joint_pos", render_mode="rgb_array",
                   reward_mode="sparse", sim_backend="cpu")
    env = RecordEpisode(env, output_dir=args.record_dir, trajectory_name=name, save_video=args.save_video,
                        source_type="motionplanning", source_desc="scripted pick-and-place for xarm6_robotiq",
                        video_fps=20, save_on_reset=False)
    h5_path = env._h5_file.filename
    target = args.num_traj // args.num_procs
    seed = args.seed_start + proc_id  # 프로세스마다 서로 다른 시드를 쓴다: proc_id, proc_id+P, proc_id+2P, …
    saved = tried = 0
    pbar = tqdm(total=target, desc=f"proc {proc_id}", position=proc_id)
    while saved < target:
        try:
            res = solve(env, seed)
        except Exception as e:  # 계획 실패 등은 그 시드를 건너뛴다
            res = -1
            if args.verbose:
                print("solve error:", repr(e))
        ok = res != -1 and bool(res[-1]["success"].item())
        tried += 1
        seed += args.num_procs
        env.flush_trajectory(save=ok)
        if args.save_video:
            env.flush_video(save=ok)
        if ok:
            saved += 1
            pbar.update(1)
            pbar.set_postfix(success_rate=round(saved / tried, 3))
    env.close()
    return h5_path, saved, tried


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num-traj", type=int, default=10)
    p.add_argument("--traj-name", default="place_cube")
    p.add_argument("--env-id", default=PLACE_ENV_ID, help="예: PlaceObjectXArm6VLA-banana-v1")
    p.add_argument("--record-dir", default="raw_place")
    p.add_argument("--num-procs", type=int, default=1)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.num_procs > 1:
        with mp.Pool(args.num_procs) as pool:
            res = pool.starmap(_main, [(args, i) for i in range(args.num_procs)])
        out = osp.join(args.record_dir, args.traj_name + ".h5")
        merge_trajectories(out, [r[0] for r in res])
        for path, _, _ in res:
            os.remove(path)
            os.remove(path.replace(".h5", ".json"))
    else:
        res = [_main(args)]
        out = res[0][0]
    saved, tried = sum(r[1] for r in res), sum(r[2] for r in res)
    print(f"\n저장 {saved}개 / 시도 {tried}개 (자동 시연 성공률 {saved / tried:.1%}) -> {out}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
