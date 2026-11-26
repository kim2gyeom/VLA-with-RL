#!/usr/bin/env bash
# 2단계: 네 물체를 한 정책으로 다루는 PPO 교사를 단계별로 학습한다. 각 단계는 앞 단계의 체크포인트에서 이어 간다.
# 교사는 카메라를 보지 않고 시뮬레이터 상태(관절, 물체·표시 위치, 접촉 여부 등 52개 + 물체 번호 + 직전 행동)만 본다.
# 우리 환경(GPU 1장, 약 80 GB)에서 단계별로 수 시간씩, 합쳐서 약 7.6억 스텝 걸렸다.
# 사용: bash scripts/02_train_teacher.sh T2|T3|T4|T5   (결과: runs/<단계>/latest.pt)
#   다음 단계로 넘어가기 전에, 앞 단계의 좋은 체크포인트를 runs/<다음 단계>/start.pt 로 복사해 둔다.
#   (실제로는 T2 iter 1750 -> T3, T3 iter 1950 -> T4, T4 iter 2300 -> T5 를 썼다.)
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/sim"
STAGE="${1:?T2, T3, T4, T5 중 하나}"; OUT="$ROOT/runs/$STAGE"; mkdir -p "$OUT"
BANKS="--bank-dir $ROOT/data/banks"
case "$STAGE" in
  T2) # 처음부터. 시연의 끝 부분에서 시작해 앞으로 당겨 오다가, 끝까지 오면 무작위 시작으로 바꾼다.
      python ppo_teacher.py --out "$OUT" $BANKS --env-counts mustard=5120,cube=1024,banana=1024,baseball=1024 \
        --mix 0.3 --forward-mix 0.25 --eval-every 50 --hours 10 ;;
  T3) # 그리퍼 명령이 자주 바뀌면 벌점(0.03), 행동 변화 벌점 0.002 -> 0.1
      python ppo_teacher.py --out "$OUT" $BANKS --resume "$OUT/start.pt" --num-envs 2048 --forward-mix 0.25 \
        --grip-coef 0.03 --rate-coef 0.1 --eval-every 25 --hours 12 ;;
  T4) # 팔 행동을 부드럽게(실행 = 0.3 * 정책 출력 + 0.7 * 직전 실행), 그리퍼 벌점 0.1
      python ppo_teacher.py --out "$OUT" $BANKS --resume "$OUT/start.pt" --num-envs 2048 --env-counts mustard=4096 \
        --forward-mix 0.25 --smooth 0.7 --grip-coef 0.1 --rate-coef 0.1 --eval-every 25 --hours 12 ;;
  T5) # 그리퍼를 실물처럼 천천히 닫히게: 0.2 m/s 에서 0.05 m/s(완전히 닫는 데 약 1.7초)까지, 성공률이 유지될 때마다 0.9배
      python ppo_teacher.py --out "$OUT" $BANKS --resume "$OUT/start.pt" --num-envs 2048 --env-counts mustard=4096 \
        --forward-mix 0.25 --smooth 0.7 --grip-coef 0.1 --rate-coef 0.1 --eval-every 25 \
        --gripper-speed 0.05 --gripper-speed-start 0.2 --hours 4 ;;
  *) echo "단계는 T2, T3, T4, T5 중 하나"; exit 1 ;;
esac
