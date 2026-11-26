"""집어서 놓기 과제: 빨간 큐브를 집어, 지시문이 말한 색의 사각 표시 위에 내려놓는다.

- 테이블 위에 파란 표시와 노란 표시가 무작위 위치에 하나씩 놓인다(좌우가 매번 섞인다).
- 지시문은 회차마다 "…place it on the blue square." 또는 "…on the yellow square." 중 하나다.
- 성공: 큐브가 지시된 표시 위(중심에서 3 cm 이내)에 놓여 정지해 있고, 그리퍼가 놓은 상태.
- 카메라와 로봇은 PickCubeXArm6VLA-v2 와 같다(비스듬한 외부 시점 base, 손목 hand, 위쪽 시점 top).
"""
import sapien
import torch

from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from pickcube_xarm6_env import PickCubeXArm6VLAEnvV2

PLACE_ENV_ID = "PlaceCubeXArm6VLA-v1"
TARGETS = [("blue", [0.10, 0.30, 1.00, 1]), ("yellow", [1.00, 0.85, 0.10, 1])]
TARGET_HALF = 0.04      # 표시는 한 변 8 cm 정사각형
PLACE_THRESH = 0.03     # 큐브 중심이 표시 중심에서 이 거리 안이면 '위에 놓임'
CUBE_XY = (0.10, 0.10)  # 큐브가 놓이는 영역의 반폭 (m)
TARGET_XY = (0.10, 0.16)  # 표시가 놓이는 영역의 반폭 (m)
MIN_TARGET_GAP = 0.13   # 두 표시 중심 사이의 최소 거리
MIN_CUBE_GAP = 0.09     # 큐브와 각 표시 중심 사이의 최소 거리


@register_env(PLACE_ENV_ID, max_episode_steps=300)
class PlaceCubeXArm6VLAEnv(PickCubeXArm6VLAEnvV2):
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._paint_table()
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.targets = [
            actors.build_box(
                self.scene, half_sizes=[TARGET_HALF, TARGET_HALF, 0.001], color=color, name=f"target_{name}",
                body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(p=[0, 0.2 * (i * 2 - 1), 0.001]),
            )
            for i, (name, color) in enumerate(TARGETS)
        ]

    @staticmethod
    def _sample_xy(half, avoid, gap):
        for _ in range(200):
            xy = (torch.rand(2) * 2 - 1) * torch.tensor(half)
            if all(torch.linalg.norm(xy - a) >= gap for a in avoid):
                return xy
        return xy

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            cube_xyz = torch.zeros((b, 3))
            target_xyz = torch.zeros((b, 2, 3))
            for i in range(b):
                t0 = self._sample_xy(TARGET_XY, [], 0.0)
                t1 = self._sample_xy(TARGET_XY, [t0], MIN_TARGET_GAP)
                target_xyz[i, 0, :2], target_xyz[i, 1, :2] = t0, t1
                cube_xyz[i, :2] = self._sample_xy(CUBE_XY, [t0, t1], MIN_CUBE_GAP)
            cube_xyz[:, 2] = self.cube_half_size
            target_xyz[:, :, 2] = 0.001
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, qs))
            for k, target in enumerate(self.targets):
                target.set_pose(Pose.create_from_pq(target_xyz[:, k]))
            if not hasattr(self, "goal_idx") or self.goal_idx.shape[0] != self.num_envs:
                self.goal_idx = torch.zeros(self.num_envs, dtype=torch.long)
            self.goal_idx[env_idx] = torch.randint(0, 2, (b,))

    # --- 데이터 기록·평가 코드가 쓰는 공통 인터페이스 ---
    def goal_color(self, i: int = 0) -> str:
        return TARGETS[int(self.goal_idx[i])][0]

    def task_text(self, i: int = 0) -> str:
        return f"Pick up the red cube and place it on the {self.goal_color(i)} square."

    def goal_position(self) -> torch.Tensor:
        pos = torch.stack([t.pose.p for t in self.targets], dim=1)  # (b, 2, 3)
        return pos[torch.arange(pos.shape[0]), self.goal_idx.to(pos.device)]

    def _get_obs_extra(self, info: dict):
        return dict(is_grasped=info["is_grasped"], tcp_pose=self.agent.tcp_pose.raw_pose, goal_pos=self.goal_position())

    def evaluate(self):
        cube = self.cube.pose.p
        pos = torch.stack([t.pose.p for t in self.targets], dim=1)
        dist = torch.linalg.norm(cube[:, None, :2] - pos[..., :2], dim=-1)  # (b, 2)
        rows = torch.arange(dist.shape[0])
        goal = self.goal_idx.to(dist.device)
        on_table = torch.abs(cube[:, 2] - self.cube_half_size) < 0.01
        is_grasped = self.agent.is_grasping(self.cube)
        is_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        released = on_table & is_static & ~is_grasped
        return {
            "success": (dist[rows, goal] <= PLACE_THRESH) & released,
            "on_wrong_target": (dist[rows, 1 - goal] <= TARGET_HALF) & released,
            "is_grasped": is_grasped,
            "is_robot_static": self.agent.is_static(0.2),
        }

    def compute_dense_reward(self, obs, action, info):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return info["success"].float()
