"""강화학습 교사 정책용 환경. 과제는 place_cube_env / place_object_env 와 같고, 관측만 좌표 기반으로 넓힌다.

- 환경 ID: PlaceRL-cube-v1, PlaceRL-banana-v1, PlaceRL-mustard-v1, PlaceRL-baseball-v1
- 관측(state 모드): 관절각·속도, 손끝 자세, 물체 자세, 목표 표시와 다른 표시의 위치,
  손끝→물체·물체→목표 벡터, 잡고 있는지 여부. 영상은 쓰지 않는다.
- 보상: 성공하면 1, 아니면 0 (reward_mode="sparse").
- 성공 조건은 원래 과제보다 엄격하다. 원래 조건만 쓰면 교사가 물체를 잡지 않고 밀어서 표시에 올린다(큐브에서 실제로 그랬다).
    1. 잡은 채로 LIFT_HEIGHT 이상 들어 올린 적이 있어야 한다 (lifted).
    2. 들었던 물체를 DROP_HEIGHT 보다 높은 곳에서 놓치면 그 회차는 실패로 끝난다 (fail). 던지거나 떨어뜨려 놓는 것을 막는다.
    3. 키 큰 물체(머스터드 병)가 잡히지 않은 채 쓰러지면 그 회차는 실패로 끝난다 (tipped → fail). 접근하다 쓰러뜨리는 경우와
       내려놓다 쓰러뜨리는 경우 모두 해당한다. 쓰러진 병은 시연에 없는 상태라 되살릴 수 없는데, 끝내지 않으면 남은 시간
       수백 스텝이 쓸모없는 경험으로 채워져 학습이 매우 느려진다.
- 시연의 중간 상태로 되돌려 시작할 수 있다: 상태 사전에 "goal_idx" 와 "lifted" 를 넣어 주면 함께 복원한다.
"""
import math

import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

import place_cube_env as pc
import place_object_env as po
from place_cube_env import PlaceCubeXArm6VLAEnv
from place_object_env import OBJECTS, PlaceObjectXArm6VLAEnv


LIFT_HEIGHT = 0.04  # 잡은 채 이만큼(m) 들어 올린 적이 있어야 성공으로 친다
DROP_HEIGHT = 0.03  # 들었던 물체가 잡히지 않은 채 이 높이(m) 위에 있으면 떨어뜨린 것으로 본다
DROP_STEPS = 3      # 위 상태가 이 횟수만큼 이어지면 실패 (잡힘 판정이 한두 번 깜빡이는 것은 봐준다)
TALL_OBJECT = 0.10  # 이보다 키가 큰 물체(m)는 쓰러짐을 실패로 본다
TIP_ANGLE = 0.6     # 잡히지 않은 물체가 수직에서 이만큼(rad, 약 34도) 넘게 기울면 쓰러진 것으로 본다 (시연 전체에서 최대 24.6도)


def _first_valid(cand, ok, fallback_ok=None):
    """후보 (b, K, 2) 중 조건을 만족하는 첫 번째를 고른다. 없으면 느슨한 조건(fallback_ok)의 첫 번째, 그것도 없으면 마지막 후보."""
    k = torch.full((cand.shape[0],), cand.shape[1] - 1, device=cand.device)
    if fallback_ok is not None:
        k = torch.where(fallback_ok.any(1), fallback_ok.float().argmax(1), k)
    k = torch.where(ok.any(1), ok.float().argmax(1), k)
    return cand[torch.arange(cand.shape[0], device=cand.device), k]


class _PlaceRLMixin:
    def _flags(self):
        if not hasattr(self, "lifted") or self.lifted.shape[0] != self.num_envs:
            self.lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.air_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.tip_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _height(self):
        """물체가 탁자에 놓였을 때보다 얼마나 높이 있는지 (m)."""
        rest = self._rest_z if hasattr(self, "_rest_z") else float(self.cube_half_size)
        return self.cube.pose.p[:, 2] - rest

    def evaluate(self):
        info = super().evaluate()
        self._flags()
        h, grasped = self._height(), info["is_grasped"]
        self.lifted |= grasped & (h > LIFT_HEIGHT)
        in_air = self.lifted & ~grasped & (h > DROP_HEIGHT)
        self.air_steps = torch.where(in_air, self.air_steps + 1, torch.zeros_like(self.air_steps))
        tipped = torch.zeros_like(self.lifted)
        if hasattr(self, "_rest_z") and self._hi[2] - self._lo[2] > TALL_OBJECT:
            q = self.cube.pose.q  # (w, x, y, z). 물체의 세로축과 수직선 사이 각의 코사인 = 1 - 2(x² + y²)
            down = ~grasped & (1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) < math.cos(TIP_ANGLE))
            self.tip_steps = torch.where(down, self.tip_steps + 1, torch.zeros_like(self.tip_steps))
            tipped = self.tip_steps >= DROP_STEPS
        fail = (self.air_steps >= DROP_STEPS) | tipped
        info["success"] = info["success"] & self.lifted & ~fail
        info["fail"] = fail
        info["tipped"] = tipped
        info["lifted"] = self.lifted.clone()
        return info

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """원래 과제와 같은 분포로 물체와 표시를 놓되, 수천 개 환경을 한꺼번에 계산한다."""
        self._flags()
        self.lifted[env_idx] = False
        self.air_steps[env_idx] = 0
        self.tip_steps[env_idx] = 0
        with torch.device(self.device):
            b, K = len(env_idx), 256
            self.table_scene.initialize(env_idx)
            is_object = hasattr(self, "_rest_z")
            if is_object:
                gripper = self.agent.controller.controllers["gripper_active"]
                gripper.config.force_limit = po.GRIP_FORCE_LIMIT
                gripper.set_drive_property()
            obj_half = torch.tensor(po.OBJECT_XY if is_object else pc.CUBE_XY)
            obj_gap = po.MIN_OBJECT_GAP if is_object else pc.MIN_CUBE_GAP
            tgt_half = torch.tensor(pc.TARGET_XY)
            t0 = (torch.rand(b, 2) * 2 - 1) * tgt_half
            c1 = (torch.rand(b, K, 2) * 2 - 1) * tgt_half
            t1 = _first_valid(c1, torch.linalg.norm(c1 - t0[:, None], dim=-1) >= pc.MIN_TARGET_GAP)
            co = (torch.rand(b, K, 2) * 2 - 1) * obj_half
            clear = (torch.linalg.norm(co - t0[:, None], dim=-1) >= obj_gap) & (torch.linalg.norm(co - t1[:, None], dim=-1) >= obj_gap)
            ok = clear
            if is_object and self._hi[2] - self._lo[2] > 0.10:  # 높은 물체는 그리퍼 바로 밑에 두지 않는다
                tcp_xy = self.agent.tcp.pose.p[env_idx, :2]
                ok = clear & (torch.linalg.norm(co - tcp_xy[:, None], dim=-1) >= 0.11)
            xy = _first_valid(co, ok, clear)
            yaw = torch.rand(b) * 2 * math.pi
            quat = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], 1)
            z = torch.full((b, 1), (self._rest_z + 0.002) if is_object else float(self.cube_half_size))
            self.cube.set_pose(Pose.create_from_pq(torch.cat([xy, z], 1), quat))
            for target, t in zip(self.targets, (t0, t1)):
                target.set_pose(Pose.create_from_pq(torch.cat([t, torch.full((b, 1), 0.001)], 1)))
            if not hasattr(self, "goal_idx") or self.goal_idx.shape[0] != self.num_envs:
                self.goal_idx = torch.zeros(self.num_envs, dtype=torch.long)
            self.goal_idx[env_idx] = torch.randint(0, 2, (b,))

    def _target_positions(self):
        pos = torch.stack([t.pose.p for t in self.targets], dim=1)  # (b, 2, 3)
        rows = torch.arange(pos.shape[0], device=pos.device)
        goal = self.goal_idx.to(pos.device)
        return pos[rows, goal], pos[rows, 1 - goal]

    def _get_obs_extra(self, info: dict):
        goal, other = self._target_positions()
        obj, tcp = self.cube.pose, self.agent.tcp.pose
        return dict(
            is_grasped=info["is_grasped"],
            lifted=info["lifted"],
            tcp_pose=tcp.raw_pose,
            obj_pose=obj.raw_pose,
            goal_pos=goal,
            other_pos=other,
            tcp_to_obj=obj.p - tcp.p,
            obj_to_goal=goal - obj.p,
        )

    def set_state_dict(self, state: dict, env_idx=None):
        state = dict(state)
        goal_idx = state.pop("goal_idx", None)
        lifted = state.pop("lifted", None)
        idx = slice(None) if env_idx is None else env_idx
        self._flags()  # 상태를 되돌릴 때는 _initialize_episode 가 불리지 않으므로 여기서 초기화한다
        self.lifted[idx] = False if lifted is None else lifted.to(self.lifted.device).bool()
        self.air_steps[idx] = 0
        self.tip_steps[idx] = 0
        if goal_idx is not None:
            if not hasattr(self, "goal_idx") or self.goal_idx.shape[0] != self.num_envs:
                self.goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.goal_idx[idx] = goal_idx.to(self.goal_idx.device).long()
        super().set_state_dict(state, env_idx)


RL_ENV_IDS = {}

register_env("PlaceRL-cube-v1", max_episode_steps=400)(type("PlaceRL_cube", (_PlaceRLMixin, PlaceCubeXArm6VLAEnv), {}))
RL_ENV_IDS["cube"] = "PlaceRL-cube-v1"
for _key in OBJECTS:
    register_env(f"PlaceRL-{_key}-v1", max_episode_steps=400)(
        type(f"PlaceRL_{_key}", (_PlaceRLMixin, PlaceObjectXArm6VLAEnv), {"object_key": _key})
    )
    RL_ENV_IDS[_key] = f"PlaceRL-{_key}-v1"
