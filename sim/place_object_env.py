"""생활용품을 집어서 지시된 색 표시 위에 내려놓는 과제. 큐브 대신 YCB 물체 하나가 놓인다.

- 과제 구조(파란·노란 표시, 지시문, 카메라)는 place_cube_env.py 와 같다.
- 물체마다 환경 ID 가 하나씩 등록된다: PlaceObjectXArm6VLA-<키>-v1  (키: OBJECTS 참고)
- 물체마다 위에서 집는 위치가 다르다.
    lying  : 누운 긴 물체. 길이의 가운데를, 뻗은 방향에 수직으로 잡는다 (바나나)
    top    : 세워진 높은 물체. 꼭대기에서 TOP_GRASP_DEPTH 만큼 아래를 잡는다 (병의 어깨·몸통 윗부분)
    center : 낮은 물체. 가운데 높이를 잡는다 (공)
- 성공: 물체가 지시된 표시 중심에서 4 cm 이내에 놓여 정지해 있고(쓰러지지 않고), 그리퍼가 놓은 상태.
"""
import json
from pathlib import Path

import numpy as np
import sapien
import torch
import trimesh
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill import ASSET_DIR
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from place_cube_env import MIN_TARGET_GAP, TARGET_HALF, TARGET_XY, TARGETS, PlaceCubeXArm6VLAEnv

YCB = Path(ASSET_DIR) / "assets/mani_skill2_ycb"  # ManiSkill 자산 폴더 (환경 변수 MS_ASSET_DIR 로 바꿀 수 있다)
# 키: (YCB 모델 이름, 지시문에 쓰는 이름, 집는 방식)
OBJECTS = {
    "banana": ("011_banana", "banana", "lying"),
    "mustard": ("006_mustard_bottle", "mustard bottle", "top"),
    "baseball": ("055_baseball", "baseball", "center"),
}
# 학습 시연에 없는 물체. 학습한 모델이 처음 보는 물체를 얼마나 다루는지 평가할 때만 쓴다(자동 시연용 집는 방식은 검증하지 않았다).
UNSEEN_OBJECTS = {
    "tennisball": ("056_tennis_ball", "tennis ball", "center"),
    "lemon": ("014_lemon", "lemon", "center"),
    "rubikscube": ("077_rubiks_cube", "Rubik's cube", "center"),
    "meatcan": ("010_potted_meat_can", "potted meat can", "center"),
    "soupcan": ("005_tomato_soup_can", "tomato soup can", "center"),
}
ALL_OBJECTS = {**OBJECTS, **UNSEEN_OBJECTS}
OBJECT_XY = (0.08, 0.10)   # 물체 중심이 놓이는 영역의 반폭 (m)
MIN_OBJECT_GAP = 0.12      # 물체 중심과 각 표시 중심 사이의 최소 거리
OBJECT_PLACE_THRESH = 0.04
TOP_GRASP_DEPTH = 0.05     # 'top' 방식에서 꼭대기로부터 내려가 잡는 깊이 (m)
# 그리퍼 관절의 최대 구동력. ManiSkill 기본값(0.1)은 60 g 큐브에는 충분하지만 200~800 g 물체는 들다가 놓친다.
# 실제 Robotiq 2F-85 의 파지력(20 N 이상)에 가깝도록 이 과제에서만 올린다. 큐브 과제에는 영향이 없다.
GRIP_FORCE_LIMIT = 1.0
ANGLES = np.deg2rad(np.arange(0, 180, 5))


def env_id_for(key: str) -> str:
    return f"PlaceObjectXArm6VLA-{key}-v1"


class PlaceObjectXArm6VLAEnv(PlaceCubeXArm6VLAEnv):
    object_key = "banana"

    def _load_scene(self, options: dict):
        model_id, self.object_name, self.grasp_mode = ALL_OBJECTS[self.object_key]
        scale = json.loads((YCB / "info_pick_v0.json").read_text())[model_id]["scales"][0]
        self._verts = np.asarray(trimesh.load(YCB / "models" / model_id / "collision.ply", force="mesh").vertices) * scale
        self._lo, self._hi = self._verts.min(0), self._verts.max(0)
        self._rest_z = float(-self._lo[2])  # 테이블에 놓였을 때 물체 원점의 높이

        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._paint_table()
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose(p=[0, 0, self._rest_z + 0.01])
        self.cube = builder.build(name="object")  # 기록·평가 코드가 cube 라는 이름으로 찾는다
        self.targets = [
            actors.build_box(
                self.scene, half_sizes=[TARGET_HALF, TARGET_HALF, 0.001], color=color, name=f"target_{name}",
                body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(p=[0, 0.2 * (i * 2 - 1), 0.001]),
            )
            for i, (name, color) in enumerate(TARGETS)
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            gripper = self.agent.controller.controllers["gripper_active"]
            gripper.config.force_limit = GRIP_FORCE_LIMIT
            gripper.set_drive_property()
            tcp_xy = self.agent.tcp.pose.p[0, :2].cpu()
            tall = self._hi[2] - self._lo[2] > 0.10  # 높은 물체는 시작할 때 그리퍼 바로 밑에 두지 않는다
            obj_xyz = torch.zeros((b, 3))
            quat = torch.zeros((b, 4))
            target_xyz = torch.zeros((b, 2, 3))
            for i in range(b):
                t0 = self._sample_xy(TARGET_XY, [], 0.0)
                t1 = self._sample_xy(TARGET_XY, [t0], MIN_TARGET_GAP)
                target_xyz[i, 0, :2], target_xyz[i, 1, :2] = t0, t1
                for _ in range(200):
                    xy = self._sample_xy(OBJECT_XY, [t0, t1], MIN_OBJECT_GAP)
                    if not tall or torch.linalg.norm(xy - tcp_xy) >= 0.11:
                        break
                obj_xyz[i, :2] = xy
                quat[i] = torch.tensor(euler2quat(0, 0, float(torch.rand(1)) * 2 * np.pi, "sxyz"))
            obj_xyz[:, 2] = self._rest_z + 0.002
            target_xyz[:, :, 2] = 0.001
            self.cube.set_pose(Pose.create_from_pq(obj_xyz, quat))
            for k, target in enumerate(self.targets):
                target.set_pose(Pose.create_from_pq(target_xyz[:, k]))
            if not hasattr(self, "goal_idx") or self.goal_idx.shape[0] != self.num_envs:
                self.goal_idx = torch.zeros(self.num_envs, dtype=torch.long)
            self.goal_idx[env_idx] = torch.randint(0, 2, (b,))

    def task_text(self, i: int = 0) -> str:
        return f"Pick up the {self.object_name} and place it on the {self.goal_color(i)} square."

    def object_top_z(self) -> float:
        return float(self.cube.pose.p[0, 2]) + float(self._hi[2])

    def grasp_spec(self):
        """지금 놓인 자세에서 위에서 집을 지점(세계 좌표)과 손가락이 닫히는 방향을 돌려준다."""
        v, lo, hi = self._verts, self._lo, self._hi
        if self.grasp_mode == "lying":
            axis = int(np.argmax((hi - lo)[:2]))
            s = (lo[axis] + hi[axis]) / 2
            band = v[np.abs(v[:, axis] - s) <= 0.012]
            before = v[np.abs(v[:, axis] - (s - 0.02)) <= 0.012][:, :2].mean(0)
            after = v[np.abs(v[:, axis] - (s + 0.02)) <= 0.012][:, :2].mean(0)
            along = (after - before) / np.linalg.norm(after - before)
            closing = np.array([-along[1], along[0]])
            z = (band[:, 2].max() + band[:, 2].min()) / 2
        else:
            z = hi[2] - TOP_GRASP_DEPTH if self.grasp_mode == "top" else (lo[2] + hi[2]) / 2
            band = v[np.abs(v[:, 2] - z) <= 0.012]
            dirs = np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)
            proj = band[:, :2] @ dirs.T
            closing = dirs[int((proj.max(0) - proj.min(0)).argmin())]
            along = np.array([-closing[1], closing[0]])
        a, b = band[:, :2] @ closing, band[:, :2] @ along
        xy = closing * (a.max() + a.min()) / 2 + along * (b.max() + b.min()) / 2
        pose = self.cube.pose.sp
        rot = quat2mat(pose.q)
        center = rot @ np.array([xy[0], xy[1], z]) + pose.p
        closing_w = rot @ np.array([closing[0], closing[1], 0.0])
        closing_w[2] = 0
        center[2] = max(center[2], 0.015)  # 손끝이 테이블에 닿지 않게
        return center, closing_w / np.linalg.norm(closing_w)

    def evaluate(self):
        obj = self.cube.pose.p
        pos = torch.stack([t.pose.p for t in self.targets], dim=1)
        dist = torch.linalg.norm(obj[:, None, :2] - pos[..., :2], dim=-1)
        rows = torch.arange(dist.shape[0])
        goal = self.goal_idx.to(dist.device)
        on_table = torch.abs(obj[:, 2] - self._rest_z) < 0.015  # 쓰러지면 원점 높이가 달라진다
        is_grasped = self.agent.is_grasping(self.cube)
        is_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        released = on_table & is_static & ~is_grasped
        return {
            "success": (dist[rows, goal] <= OBJECT_PLACE_THRESH) & released,
            "on_wrong_target": (dist[rows, 1 - goal] <= TARGET_HALF) & released,
            "is_grasped": is_grasped,
            "is_robot_static": self.agent.is_static(0.2),
        }


for _key in ALL_OBJECTS:
    register_env(env_id_for(_key), max_episode_steps=300)(
        type(f"PlaceObject_{_key}", (PlaceObjectXArm6VLAEnv,), {"object_key": _key})
    )
