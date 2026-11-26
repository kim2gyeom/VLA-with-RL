# xArm6 놓기 과제 강화학습 환경 (ManiSkill3)

*English summary: ManiSkill3 environments for an xArm6 (Robotiq gripper, RealSense D435 wrist camera) that picks one object (cube, banana, mustard bottle, baseball) and places it on an instructed blue or yellow square. Includes a state-based multi-object PPO teacher with a reverse curriculum from motion-planner demos, a gripper closing-speed limit, desk / lighting / camera randomization, and scripts that record camera demonstrations from the teacher and convert them to LeRobot datasets for VLA (π₀.₅) fine-tuning.*

사람이 조종한 시연 없이 VLA(π₀.₅)를 학습시키기 위해 만든 시뮬레이션 환경과 강화학습 교사 코드다. 흐름은 세 단계다.

1. 모션 플래너로 자동 시연을 만들고, 그 시뮬레이터 상태를 묶는다.
2. 이 묶음을 써서 PPO 교사를 학습한다. 교사는 카메라를 보지 않고 시뮬레이터 상태만 본다.
3. 학습된 교사를 카메라가 달린 환경에서 돌려 영상 시연을 기록하고, LeRobot 데이터셋으로 바꾼다.

## 과제

- 로봇: xArm6 + Robotiq 2F-85 (ManiSkill `xarm6_robotiq`). 관절 위치 제어(`pd_joint_pos`), 20 Hz.
- 책상 위에 물체 하나와 색 표시 두 개(파란색, 노란색, 한 변 8 cm)가 무작위로 놓인다.
- 지시문 예: `Pick up the banana and place it on the blue square.`
- 성공: 물체가 지시된 표시 중심에서 4 cm 이내에 쓰러지지 않고 놓여 정지해 있고, 그리퍼가 놓은 상태.
- 학습용 물체: 큐브(빨강), 바나나, 머스터드 병, 야구공. 평가용 처음 보는 물체: 테니스공, 레몬, 루빅스 큐브, 고기 통조림, 토마토 수프 캔.
- 카메라:
  - 외부 `base_camera`: 256×256, 로봇 받침 기준 앞 97 cm, 왼쪽 55 cm, 책상 위 45 cm에서 작업 영역을 비스듬히 본다. 화각 52°.
  - 손목 `hand_camera`: `wrist_camera="d435"`면 UFACTORY 순정 거치대에 단 RealSense D435(320×240, fx = fy = 307.5). 장착 위치는 xarm_ros2 의 `realsense_d435i.urdf.xacro`에서 가져왔다.

## 파일

| 파일 | 내용 |
|---|---|
| `sim/pickcube_xarm6_env.py` | 바탕 환경. 손목 카메라, 그리퍼 속도 제한, 책상 색, 조명·외부 카메라 바꾸기 |
| `sim/place_cube_env.py` | 큐브를 색 표시 위에 놓는 과제 (`PlaceCubeXArm6VLA-v1`) |
| `sim/place_object_env.py` | YCB 물체 과제 (`PlaceObjectXArm6VLA-<물체>-v1`). 물체별로 위에서 집을 위치를 정한다 |
| `sim/rl_place_env.py` | 강화학습용 환경 (`PlaceRL-<물체>-v1`). 보상, 상태 관측, 시연 상태에서 시작하기 |
| `sim/gen_place_demos.py` | 모션 플래너 자동 시연 (성공한 것만 저장) |
| `sim/build_demo_bank.py` | 자동 시연의 시뮬레이터 상태를 한 파일(bank)로 묶음 |
| `sim/ppo_teacher.py` | 네 물체를 한 정책으로 다루는 PPO 교사. GPU 병렬 환경 |
| `sim/teacher_rollout.py` | 학습된 교사를 카메라 환경에서 돌려 영상 시연을 HDF5로 기록 |
| `sim/replay_to_h5.py` | 공용 함수(그리퍼 값, 좌표 변환)와 자동 시연 영상 재기록 |
| `sim/h5_to_lerobot.py`, `sim/merge_place4.py` | HDF5 → LeRobot 데이터셋 변환, 물체별 데이터셋 합치기 |
| `scripts/01_scripted_demos_and_banks.sh` | 1단계 |
| `scripts/02_train_teacher.sh` | 2단계 (T2 → T3 → T4 → T5) |
| `scripts/03_record_demos.sh` | 3단계 |
| `data/gripper_calibration.json` | 그리퍼 관절각과 손가락 간격의 실측표 |

## 설치

확인한 버전: Python 3.11, `mani_skill` 3.0.1, `sapien` 3.0.3, `torch` 2.9.1 (CUDA 12.8), `gymnasium` 1.4.0. 데이터셋 변환은 `lerobot` 0.6.1을 쓴다.

```bash
conda create -n maniskill python=3.11 && conda activate maniskill
pip install mani_skill==3.0.1 torch h5py trimesh tqdm "numpy<2"   # numpy 2 에서는 모션 플래너가 멈춘다
python -m mani_skill.utils.download_asset ycb                    # YCB 물체 (ManiSkill 자산 폴더로 내려받는다)
# 데이터셋 변환용 환경은 따로 만든다: pip install lerobot==0.6.1  (환경 이름은 LEROBOT_ENV 로 지정, 기본 lerobot)
```

모든 명령은 저장소 폴더에서 실행한다. GPU는 `CUDA_VISIBLE_DEVICES`로 고른다.

## 사용법

### 1. 자동 시연과 시연 상태 묶음

```bash
bash scripts/01_scripted_demos_and_banks.sh     # data/raw_<물체>/, data/banks/bank_<물체>.npz
```

### 2. PPO 교사 학습

```bash
bash scripts/02_train_teacher.sh T2
mkdir -p runs/T3 && cp runs/T2/latest.pt runs/T3/start.pt   # 앞 단계의 좋은 체크포인트에서 이어 간다
bash scripts/02_train_teacher.sh T3
# ... T4, T5 도 같은 방식
```

| 단계 | 바꾼 것 | 비고 |
|---|---|---|
| T2 | 처음부터. 시연의 끝 부분에서 시작해 점점 앞으로 당겨 오고(뒤에서부터 배우기), 끝까지 오면 무작위 시작으로 바꾼다 | 머스터드는 어려워 환경을 5,120개 배정 |
| T3 | 그리퍼 명령이 바뀔 때마다 벌점 0.03, 행동 변화 벌점 0.002 → 0.1 | 열었다 닫았다 반복을 줄인다 |
| T4 | 팔 행동 저역 통과(실행 = 0.3 × 정책 + 0.7 × 직전), 그리퍼 벌점 0.1 | 움직임을 부드럽게 |
| T5 | 그리퍼 닫힘 속도를 0.2 → 0.05 m/s로 단계적으로 늦춤 | 실물처럼 닫는 데 약 1.7초 |

교사가 보는 것(63개): 관절 각도·속도(각 12), 그리퍼 끝 자세(7), 물체 자세(7), 가야 할 표시와 다른 표시 위치(각 3), 그리퍼→물체와 물체→표시 벡터(각 3), 잡았나·들었나(2), 물체 번호(4), 직전 행동(7). 행동(7개)은 팔 관절 증분(최대 0.03 rad/스텝) 6개와 그리퍼 열기/닫기다.

T5의 평가 성공률(무작위 시작): 큐브 99.0%, 바나나 96.7%, 머스터드 87.5%, 야구공 99.7%.

### 3. 영상 시연 기록과 LeRobot 변환

```bash
# 기본 화면 (나무 책상, 고정 조명·카메라)
PER=150 NOISE=0.5 MAX_TOGGLES=4 bash scripts/03_record_demos.sh runs/T5/latest.pt teach4g
# 흰 책상 중심으로 책상·조명·외부 카메라를 시연마다 바꿔 가며
VISUAL_RAND=1.5 TABLE_COLOR=white_mix PER=150 NOISE=0.5 MAX_TOGGLES=4 bash scripts/03_record_demos.sh runs/T5/latest.pt teach4gv
```

- 물체마다 8묶음 × `PER`회 시도하고, 성공한 시도만 저장한다.
- 시도의 절반은 팔을 가끔 부드럽게 밀어서(세기 `NOISE`) 궤도에서 벗어났다가 돌아오는 장면을 만든다. 행동 라벨은 밀기 전 교사 행동이다.
- 그리퍼 명령이 `MAX_TOGGLES`번보다 많이 바뀐 시도는 버린다.
- 결과: `data/sim/<이름>_xarm6_d435` (외부 + D435 손목), `data/sim/<이름>_xarm6` (외부 + 기본 손목 카메라). 물체마다 12개를 점검용으로 `meta/holdout.json`에 적어 둔다.
- `--video` 로 mp4를 함께 저장하려면 `ffmpeg`가 PATH에 있거나 환경 변수 `FFMPEG`에 경로를 준다.

## 환경 옵션 (`gym.make` 인자)

| 인자 | 기본값 | 내용 |
|---|---|---|
| `wrist_camera` | `generic` | `generic`, `d435`, `both` |
| `gripper_speed` | 0 | 손가락 간격이 바뀌는 최대 속도(m/s). 0이면 제한 없음(약 0.1초에 닫힘). 0.05면 약 1.7초 |
| `table_color` | 없음(나무 무늬) | `white`, `(r, g, b)`, 또는 `white_mix`(흰색 계열 2/3, 나무 1/6, 채도 낮은 다른 색 1/6을 회차마다 고름) |
| `visual_rand` | 0 | 화면 바꾸기 세기. 1이면 조명 밝기 ±50~60%, 색감 ±15%, 방향 무작위, 외부 카메라 위치 ±3 cm, 바라보는 점 ±2 cm, 화각 ±5% |

`table_color="white_mix"`나 `visual_rand > 0`을 쓸 때는 `reconfiguration_freq=1`을 함께 줘야 회차마다 바뀐다. 화면 바꾸기용 난수는 물체 배치용 난수와 따로 써서, 같은 시드면 물체 배치는 같다.

```python
import gymnasium as gym
import sys; sys.path.insert(0, "sim")
import rl_place_env  # 환경 등록 (place_cube_env, place_object_env 도 함께 등록된다)

env = gym.make("PlaceObjectXArm6VLA-banana-v1", obs_mode="rgb", control_mode="pd_joint_pos", sim_backend="cpu",
               wrist_camera="d435", gripper_speed=0.05, table_color="white_mix", visual_rand=1.0, reconfiguration_freq=1)
obs, _ = env.reset(seed=0)
print(env.unwrapped.task_text())
```

## 알아 둘 점

- 시뮬레이션 그리퍼는 Robotiq 2F-85 모양이다. UFACTORY xArm Gripper와 손가락 모양이 다르다.
- 그리퍼 속도 제한은 관절각을 일정 속도로 움직이는 근사다. 가속, 힘 제어, 닿은 뒤의 동작은 흉내 내지 않는다.
- 그리퍼 구동력은 YCB 물체 과제에서 ManiSkill 기본값(0.1)보다 높게(1.0) 잡았다. 200~800 g 물체를 들다 놓치지 않게 하려는 것이다.
- 교사는 화면 대신 정확한 상태를 보고, 가야 할 표시의 좌표를 직접 받는다. 그래서 실물에는 교사가 아니라 교사 시연으로 학습한 VLA를 올린다.
- 장면에는 물체가 항상 하나다. 여러 물체 중 지시된 것을 고르는 과제는 아직 없다.
- 외부 카메라 위치와 책상 색(흰색 기준값)은 우리 연구실을 기준으로 정했다. 다른 환경에서는 `OBLIQUE_EYE`, `OBLIQUE_TARGET`, `OBLIQUE_FOV`, `TABLE_COLORS`를 맞춰 쓴다.
