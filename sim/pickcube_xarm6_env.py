"""PickCube-v1 을 영상 기반 학습용으로 조금 바꾼 환경.

원본과 다른 점
- 로봇은 xarm6_robotiq 로 고정한다.
- 손목 카메라(hand_camera)를 추가해 외부(base_camera)와 두 시점을 만든다.
- 목표 위치(초록 구)를 카메라 영상에 보이게 한다. 원본은 영상에서 숨기고 숫자로만 준다.
- 한 회의 최대 길이를 200 스텝으로 늘린다. 자동 시연이 50 스텝보다 길기 때문이다.

물리, 큐브·목표의 무작위 배치, 성공 판정은 원본 그대로다.
"""
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

ENV_ID = "PickCubeXArm6VLA-v1"
TASK_TEXT = "Pick up the red cube and move it to the green goal."
IMAGE_SIZE = 256


# ---------------------------------------------------------------------------
# 손목 카메라
#   generic : ManiSkill 이 xArm6 손목 카메라용으로 정해 둔 가상 카메라(정사각형, 화각 90도)
#   d435    : RealSense D435 를 UFACTORY 순정 카메라 거치대에 단 구성
#     - 장착 위치는 UFACTORY 의 로봇 모델에서 가져왔다 (xarm_ros2: xarm_description/urdf/camera/
#       realsense_d435i.urdf.xacro). 플랜지(link_eef) 기준 camera_link = xyz (0.06746, -0.0175, 0.0237),
#       rpy (pi, -pi/2, 0), 컬러 센서는 camera_link 에서 y 로 +0.015. 카메라는 공구 축과 나란히 앞을 본다.
#     - 렌즈는 D435 컬러 스트림 640x480 의 대표값(fx = fy = 615, 중심 320/240)을 절반 해상도 320x240 으로 쓴다.
#       실제 값은 기기마다 조금 다르므로 pyrealsense2 로 읽어 D435_INTRINSIC_640 을 바꾸면 된다.
# ---------------------------------------------------------------------------
D435_MOUNT = sapien.Pose(p=[0.06746, -0.0175, 0.0237], q=euler2quat(np.pi, -np.pi / 2, 0, "sxyz")) * sapien.Pose(p=[0, 0.015, 0])
D435_INTRINSIC_640 = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]])
D435_SIZE = (320, 240)
GRIPPER_JOINT_CLOSED = 0.81  # 완전히 닫힌 그리퍼 관절각(rad)
GRIPPER_STROKE_M = 0.0859    # 완전 열림에서 닫힘까지 손가락 간격 변화(m). data/gripper_calibration.json
# 책상 윗면 색. None(기본)은 ManiSkill 의 나무 무늬 그대로. 실물 연구실 책상은 흰색이다.
# 광택 정도는 아직 모른다. 무광에 가깝게 두었다.
TABLE_COLORS = {"wood": None, "white": (0.90, 0.90, 0.88)}
TABLE_ROUGHNESS = 0.7


@register_env(ENV_ID, max_episode_steps=200)
class PickCubeXArm6VLAEnv(PickCubeEnv):
    def __init__(self, *args, robot_uids="xarm6_robotiq", wrist_camera="generic", gripper_speed=0.0, visual_rand=0.0, table_color=None, **kwargs):
        self.wrist_camera = wrist_camera
        # table_color: 책상 윗면 색. None/"wood" 는 원래 나무 무늬, "white" 는 흰색, (r, g, b) 0~1, 또는 "white_mix"(_paint_table 참고).
        self.table_color = table_color
        # visual_rand: 화면 바꾸기의 세기(0 이면 끔, 1 이 기준). 조명의 밝기·색·방향과 외부 카메라의 위치·화각을 회차마다 바꾼다.
        # 장면을 다시 만들 때만 적용되므로 gym.make 에 reconfiguration_freq=1 을 함께 준다. 물체 배치(물리 난수)는 건드리지 않는다.
        self.visual_rand = float(visual_rand)
        # gripper_speed: 손가락 간격이 바뀌는 최대 속도(m/s). 0 이면 제한 없음(기존 동작: 0.05~0.10 초 만에 닫힘).
        # 실물 그리퍼는 명령을 받아도 천천히 닫히므로, 환경이 그 느린 하드웨어를 흉내 낸다. 정책이 내는 명령(열기/닫기)은 그대로다.
        self.gripper_speed = float(gripper_speed)
        self._grip_cmd = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # --- 화면 바꾸기 (조명, 외부 카메라) ---
    def _visual_rng(self, salt: int):
        """회차 시드에서 따로 만든 난수. 물체 배치에 쓰는 난수와 섞이지 않는다."""
        return np.random.RandomState((int(self._episode_seed[0]) * 7919 + salt) % (2**31))

    def _load_lighting(self, options: dict):
        s = self.visual_rand
        if s <= 0:
            return super()._load_lighting(options)
        r = self._visual_rng(1)
        tint = lambda: 1.0 + r.uniform(-0.15, 0.15, 3) * s  # 색온도 차이를 흉내 내는 채널별 배율
        self.scene.set_ambient_light((0.3 * (1 + r.uniform(-0.6, 0.6) * s) * tint()).tolist())
        az = r.uniform(0, 2 * np.pi)
        main_dir = [float(np.cos(az)), float(np.sin(az)), -float(r.uniform(0.6, 1.8))]
        self.scene.add_directional_light(main_dir, ((1 + r.uniform(-0.5, 0.5) * s) * tint()).tolist(),
                                         shadow=self.enable_shadow, shadow_scale=5, shadow_map_size=2048)
        self.scene.add_directional_light([0, 0, -1], ((1 + r.uniform(-0.5, 0.5) * s) * tint()).tolist())

    def _jitter_camera(self, eye, target, fov):
        """외부 카메라의 위치(±3 cm), 바라보는 점(±2 cm), 화각(±5%)을 세기에 비례해 흔든다."""
        s = self.visual_rand
        if s <= 0 or getattr(self, "_episode_seed", None) is None:
            return eye, target, fov
        r = self._visual_rng(2)
        eye = (np.asarray(eye, dtype=float) + r.uniform(-0.03, 0.03, 3) * s).tolist()
        target = (np.asarray(target, dtype=float) + r.uniform(-0.02, 0.02, 3) * s).tolist()
        return eye, target, float(fov * (1 + r.uniform(-0.05, 0.05) * s))

    def _paint_table(self):
        """책상의 무늬를 지우고 한 가지 색으로 칠한다. 장면을 만들 때(_load_scene) 부른다.
        table_color="white"     : 흰색. visual_rand 가 켜져 있으면 밝기(-12%~+5%)와 색조(채널별 ±4%)를 회차마다 조금 바꾼다.
        table_color="white_mix" : 시연 기록용. 회차마다(reconfiguration_freq=1 필요) 책상을 고른다.
            2/3 흰색 계열 (밝기 0.80~1.05배, 색조 채널별 ±5%, 거칠기 0.45~0.9 — 실물 광택을 아직 몰라서 넓게. 0.4 아래는 손목 카메라에 반사광이 화면 절반을 덮는다)
            1/6 원래 나무 무늬
            1/6 채도가 낮은 아무 색 (채도 0.35 이하라 파란·노란 표시, 빨간 큐브와 헷갈리지 않는다)
        고른 결과는 self.table_desc 에 남는다(기록 파일에 회차별로 저장)."""
        self.table_desc = "wood"
        c = self.table_color
        rough = TABLE_ROUGHNESS
        if c == "white_mix":
            r = self._visual_rng(3)
            u = r.random_sample()
            if u < 2 / 3:
                rgb = np.clip(np.asarray(TABLE_COLORS["white"]) * r.uniform(0.80, 1.05) * (1 + r.uniform(-0.05, 0.05, 3)), 0, 1)
                rough = float(r.uniform(0.45, 0.9))
                kind = "white"
            elif u < 5 / 6:
                return
            else:
                import colorsys
                rgb = np.asarray(colorsys.hsv_to_rgb(r.random_sample(), r.uniform(0.0, 0.35), r.uniform(0.25, 0.9)))
                rough = float(r.uniform(0.4, 1.0))
                kind = "color"
        else:
            c = TABLE_COLORS.get(c, c) if isinstance(c, str) else c
            if c is None:
                return
            rgb = np.asarray(c, dtype=float)
            s = self.visual_rand
            if s > 0 and getattr(self, "_episode_seed", None) is not None:
                r = self._visual_rng(3)
                rgb = np.clip(rgb * (1 + r.uniform(-0.12, 0.05) * s) * (1 + r.uniform(-0.04, 0.04, 3) * s), 0, 1)
            kind = "fixed"
        for obj in self.table_scene.table._objs:
            for shape in obj.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
                for part in shape.parts:
                    m = part.material
                    m.set_base_color([*rgb.tolist(), 1.0])
                    m.set_base_color_texture(None)
                    m.set_normal_texture(None)
                    m.set_roughness_texture(None)
                    m.set_roughness(rough)
        self.table_desc = f"{kind} rgb={np.round(rgb, 3).tolist()} rough={rough:.2f}"

    # --- 그리퍼 속도 제한 ---
    def _grip_qpos(self):
        return self.agent.robot.get_qpos()[:, 6].clone()

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        # 다시 시작한 환경은 그리퍼 목표를 지금 실제 위치로 맞춘다 (잡은 채로 시작하는 상태 복원도 여기로 온다)
        q = self._grip_qpos()
        idx = (kwargs.get("options") or {}).get("env_idx", None)
        if self._grip_cmd is None or self._grip_cmd.shape != q.shape or idx is None:
            self._grip_cmd = q
        else:
            self._grip_cmd[idx] = q[idx]
        return out

    def step(self, action):
        if self.gripper_speed > 0 and action is not None:
            a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1).clone()
            if self._grip_cmd is None or self._grip_cmd.shape[0] != self.num_envs:
                self._grip_cmd = self._grip_qpos()
            # 간격 85.9 mm 가 관절각 0.81 rad 에 거의 비례한다(실측표 기준 구간별 2.6~3.4 mm / 0.03 rad)
            step = self.gripper_speed / GRIPPER_STROKE_M * GRIPPER_JOINT_CLOSED / self.control_freq
            want = a[:, 6].clamp(0.0, GRIPPER_JOINT_CLOSED)
            self._grip_cmd = self._grip_cmd + (want - self._grip_cmd).clamp(-step, step)
            a[:, 6] = self._grip_cmd
            action = a
        return super().step(action)

    def _wrist_camera_configs(self):
        """wrist_camera: generic | d435 | both. both 는 기록용으로 두 카메라를 함께 단다(hand_camera, d435_camera)."""
        cams = []
        if self.wrist_camera in ("generic", "both"):
            cams.append(self._generic_wrist_camera())
        if self.wrist_camera in ("d435", "both"):
            w, h = D435_SIZE
            k = D435_INTRINSIC_640.copy()
            k[:2] *= w / 640
            uid = "hand_camera" if self.wrist_camera == "d435" else "d435_camera"
            cams.append(CameraConfig(uid=uid, pose=D435_MOUNT, width=w, height=h, intrinsic=k,
                                     near=0.01, far=100, mount=self.agent.robot.links_map["link6"]))
        return cams

    def _generic_wrist_camera(self):
        return CameraConfig(
            uid="hand_camera",
            pose=sapien.Pose(p=[0, 0, -0.05], q=[0.70710678, 0, 0.70710678, 0]),
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            fov=np.pi / 2,
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["camera_link"],
        )

    @property
    def _default_sensor_configs(self):
        base = super()._default_sensor_configs[0]
        base.width = base.height = IMAGE_SIZE
        return [base] + self._wrist_camera_configs()

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._paint_table()
        if self.goal_site in self._hidden_objects:
            self._hidden_objects.remove(self.goal_site)

    # --- 데이터 기록·평가 코드가 쓰는 공통 인터페이스 ---
    def task_text(self, i: int = 0) -> str:
        return TASK_TEXT

    def goal_position(self):
        return self.goal_site.pose.p


# ---------------------------------------------------------------------------
# v2: 외부 카메라를 비스듬한 시점으로 바꾼 환경. 물리와 과제는 v1 과 같다.
# 다른 연구(RDT 벤치마크)가 PickCube 입력으로 쓴 것과 같은 방향의 옆-위 시점이다. 그 원래 위치
# (eye [0.6, 0.7, 0.6] -> target [0, 0, 0.35], 512 px)를 256 px 에 그대로 쓰면 큐브가 5 px 정도로
# 작아져서, 같은 방향을 유지한 채 작업 영역이 화면에 차도록 당겨 놓았다.
# v1 의 뒤쪽-위 시점은 top_camera 로 남겨서 필요하면 세 번째 시점으로 쓸 수 있다.
# ---------------------------------------------------------------------------
ENV_ID_V2 = "PickCubeXArm6VLA-v2"
OBLIQUE_EYE = [0.45, 0.55, 0.45]
OBLIQUE_TARGET = [-0.05, 0.0, 0.12]
OBLIQUE_FOV = 0.9


@register_env(ENV_ID_V2, max_episode_steps=200)
class PickCubeXArm6VLAEnvV2(PickCubeXArm6VLAEnv):
    @property
    def _default_sensor_configs(self):
        top, *wrist = super()._default_sensor_configs
        top.uid = "top_camera"
        eye, target, fov = self._jitter_camera(OBLIQUE_EYE, OBLIQUE_TARGET, OBLIQUE_FOV)
        pose = sapien_utils.look_at(eye=eye, target=target)
        base = CameraConfig("base_camera", pose, IMAGE_SIZE, IMAGE_SIZE, fov, 0.01, 100)
        return [base, *wrist, top]
