#!/usr/bin/env bash
# 1단계: 모션 플래너로 물체별 자동 시연을 만들고(성공한 것만), 그 시뮬레이터 상태를 "시연 상태 묶음(bank)"으로 모은다.
# 강화학습 교사는 처음에 이 시연의 끝 부분에서 시작해 점점 앞으로 당겨 오며 배운다(뒤에서부터 배우기).
# 사용: bash scripts/01_scripted_demos_and_banks.sh            (결과: data/raw_<물체>/, data/banks/bank_<물체>.npz)
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/sim"
N="${N:-1008}"; PROCS="${PROCS:-12}"
mkdir -p "$ROOT/data/banks"
env_of() { [ "$1" = cube ] && echo "PlaceCubeXArm6VLA-v1" || echo "PlaceObjectXArm6VLA-$1-v1"; }
for k in cube banana mustard baseball; do
  [ -f "$ROOT/data/raw_$k/$k.h5" ] || python gen_place_demos.py -n "$N" --num-procs "$PROCS" \
      --env-id "$(env_of $k)" --traj-name "$k" --record-dir "$ROOT/data/raw_$k" > "$ROOT/data/gen_$k.log" 2>&1 &
done
wait
for k in cube banana mustard baseball; do
  [ -f "$ROOT/data/banks/bank_$k.npz" ] || python build_demo_bank.py --raw "$ROOT/data/raw_$k/$k.h5" --out "$ROOT/data/banks/bank_$k.npz" \
      > "$ROOT/data/banks/bank_$k.log" 2>&1 &
done
wait
ls -la "$ROOT/data/banks"
