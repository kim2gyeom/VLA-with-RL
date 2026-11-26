#!/usr/bin/env bash
# PPO 교사(ppo_teacher.py)를 카메라 환경에서 실행해 네 물체의 "집어서 색 표시 위에 놓기" 시연 데이터셋을 만든다.
#   ① 물체별로 교사를 실행해 영상과 함께 기록(성공한 회차만)  ② LeRobot 변환  ③ 하나로 합치기
# 사용: bash scripts/03_record_demos.sh runs/T5/latest.pt [이름=teach4]
# 결과: data/sim/<이름>_xarm6       (외부 비스듬한 시점 + 기존 손목 카메라)
#       data/sim/<이름>_xarm6_d435  (외부 비스듬한 시점 + D435 손목 카메라)
# 예 (흰 책상 중심으로 화면을 바꿔 가며): VISUAL_RAND=1.5 TABLE_COLOR=white_mix PER=150 NOISE=0.5 MAX_TOGGLES=4 bash scripts/03_record_demos.sh runs/T5/latest.pt teach4gv
# 회차의 절반은 팔을 가끔 부드럽게 밀면서 실행하고(NOISE = 미는 세기), 기록은 밀지 않은 행동으로 한다.
# MAX_TOGGLES 를 주면 그리퍼 명령이 그보다 많이 바뀐 회차는 저장하지 않는다(정상은 2회, 다시 잡으면 4회).
# VISUAL_RAND(조명·외부 카메라 바꾸기 세기)와 TABLE_COLOR(white, white_mix 등)를 주면 화면을 회차마다 바꿔 가며 기록한다.
# 물체마다 교사가 완성되는 시점이 다르면 나눠서 돌릴 수 있다. 이미 기록한 조각(.done)은 다시 하지 않는다.
#   OBJS="cube banana baseball" ONLY_ROLLOUT=1 bash scripts/03_record_demos.sh <ckpt>   # ① 만, 세 물체
#   OBJS="" CONVERT_OBJS="cube banana baseball" ONLY_CONVERT=1 bash scripts/03_record_demos.sh <ckpt>  # ② 만, 세 물체
#   bash scripts/03_record_demos.sh <새 ckpt>                                            # 남은 물체 ① + ②③
set -eo pipefail
CKPT="$1"; NAME="${2:-teach4}"
HERE="$(cd "$(dirname "$0")" && pwd)"; SIM="$HERE/../sim"; CKPT="$(realpath "$CKPT")"
R="${DATA_ROOT:-$HERE/../data/sim}"; W="$R/${NAME}_work"; GPU="${CUDA_VISIBLE_DEVICES:-0}"
SHARDS="${SHARDS:-8}"; PER="${PER:-135}"; NOISE="${NOISE:-0.5}"; NOISE_FRAC="${NOISE_FRAC:-0.5}"; MAX_TOGGLES="${MAX_TOGGLES:-0}"
OBJS="${OBJS-cube banana mustard baseball}"   # OBJS="" 이면 ① 을 건너뛴다 (:- 로 쓰면 빈 값이 전체로 바뀌어 버린다)
mkdir -p "$W"; cd "$SIM"
source "$(conda info --base)/etc/profile.d/conda.sh"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$W/make.log"; }

conda activate "${SIM_ENV:-maniskill}"
say "① 교사 실행·기록 시작 (물체당 ${SHARDS}개 × ${PER}회차, 미는 세기 $NOISE, 미는 회차 비율 $NOISE_FRAC, 그리퍼 명령 바뀜 상한 $MAX_TOGGLES)"
for k in $OBJS; do
  for i in $(seq 0 $(( SHARDS - 1 ))); do
    [ -f "$W/$k.$i.done" ] || ( CUDA_VISIBLE_DEVICES="$GPU" python teacher_rollout.py --ckpt "$CKPT" --object "$k" --out "$W/$k.$i.h5" \
        --episodes "$PER" --seed-start $(( 200000 + i * 1000 )) --noise "$NOISE" --noise-frac "$NOISE_FRAC" --max-toggles "$MAX_TOGGLES" --wrist-camera both \
        --visual-rand "${VISUAL_RAND:-0}" ${TABLE_COLOR:+--table-color "$TABLE_COLOR"} \
        > "$W/rollout_$k.$i.log" 2>&1 && touch "$W/$k.$i.done" ) &
  done
done
wait
for k in $OBJS; do
  say "  $k: $(grep -a -h '성공' "$W"/rollout_$k.*.log | awk '{split($4, a, "/"); s+=a[1]; n+=a[2]} END {printf "성공 %d/%d (%.1f%%)", s, n, 100*s/n}')"
done
[ -n "$ONLY_ROLLOUT" ] && { say "① 까지만 하고 끝냅니다 ($OBJS)"; exit 0; }
for k in ${CONVERT_OBJS:-cube banana mustard baseball}; do
  for i in $(seq 0 $(( SHARDS - 1 ))); do [ -f "$W/$k.$i.done" ] || { say "기록이 없는 조각: $k.$i — 중단"; exit 1; }; done
done

conda activate "${LEROBOT_ENV:-lerobot}"   # LeRobot 변환용 환경
for variant in "${NAME}_xarm6|base,hand" "${NAME}_xarm6_d435|base,hand:d435"; do
  name="${variant%%|*}"; cams="${variant##*|}"
  say "② 변환 시작: $name ($cams)"
  for k in ${CONVERT_OBJS:-cube banana mustard baseball}; do
    for i in $(seq 0 $(( SHARDS - 1 ))); do
      [ -f "$W/conv_${name}_$k.$i.done" ] || ( python h5_to_lerobot.py --h5 "$W/$k.$i.h5" --root "$W/${name}__$k.$i" --repo-id "local/${name}__$k.$i" \
          --cameras "$cams" --fast --overwrite > "$W/conv_${name}_$k.$i.log" 2>&1 && touch "$W/conv_${name}_$k.$i.done" ) &
    done
  done
  wait
  [ -n "$ONLY_CONVERT" ] && continue
  say "③ 합치기: $name"
  python "$SIM/merge_place4.py" --work "$W" --name "$name" --shards "$SHARDS" --out "$R/$name" 2>&1 | grep -a -v -i warn | tail -3 | tee -a "$W/make.log"
done
say "완료"
