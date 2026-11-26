"""좌표를 직접 보는 교사 정책을 PPO 로 학습한다. 여러 물체를 한 네트워크로 배운다. 시연은 "어디서 시작할지"를 정하는 데만 쓴다.

실행 환경: conda activate maniskill
방법 (DiscoDemo 논문의 '뒤에서부터 배우기'를 PPO 에 적용)
  1. 회차를 시연의 끝에서 c 스텝 전 상태로 되돌려 시작한다. 처음에는 c 가 작아 마지막 구간만 풀면 된다.
  2. 경계(c)에서 시작한 회차의 성공률이 기준을 넘으면 c 를 늘려 더 앞에서 시작하게 한다. 물체마다 따로 센다.
  3. c 가 시연 길이에 닿으면, 그 물체는 과제의 원래 무작위 시작으로 바꿔 계속 학습한다.
보상: 성공하면 1 (rl_place_env 의 조건: 들어 올린 적이 있고, 목표 표시 위에 놓고 손을 뗌). 떨어뜨리면 0 으로 회차가 끝난다.
      행동이 급하게 바뀌면 벌점(--rate-coef), 그리퍼 명령(열기/닫기)을 바꿀 때마다 벌점(--grip-coef)을 준다.
      그리퍼 벌점이 없으면 교사가 열기/닫기를 빠르게 번갈아 내서 쥐는 힘을 조절하는 요령을 배운다(T2 에서 실제로 그랬다).
      시연의 행동은 입력에도 보상에도 쓰지 않는다.
행동: 팔 관절각 증분 6개(-1~1, 1 = --arm-step rad) + 그리퍼 1개(양수 = 닫기, 음수 = 열기). 20 Hz.
      환경에는 관절 목표각(현재 관절각 + 증분)과 그리퍼 열림/닫힘 명령으로 바꿔 넣는다. 자동 시연과 같은 제어 방식이다.
관측: 환경의 좌표 관측 + 물체 종류(원-핫) + 직전 행동.
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rl_place_env import LIFT_HEIGHT, RL_ENV_IDS

GRIPPER_CLOSED = 0.81  # 그리퍼 관절의 닫힘 명령값(rad)
OBJECT_ACTOR = {"cube": "cube"}  # 시연 상태에서 물체의 이름. 나머지 물체는 "object"


class RunningNorm:
    def __init__(self, dim, device):
        self.mean, self.var, self.count = torch.zeros(dim, device=device), torch.ones(dim, device=device), 1e-4

    def update(self, x):
        b_mean, b_var, n = x.mean(0), x.var(0), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + n
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def __call__(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10, 10)


def mlp(inp, out, hidden, out_gain=1.0):
    layers, d = [], inp
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, 2**0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        d = h
    lin = nn.Linear(d, out)
    nn.init.orthogonal_(lin.weight, out_gain)
    nn.init.zeros_(lin.bias)
    return nn.Sequential(*layers, lin)


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512, 512, 256)):
        super().__init__()
        self.actor, self.critic = mlp(obs_dim, act_dim, hidden, out_gain=0.01), mlp(obs_dim, 1, hidden)
        self.logstd = nn.Parameter(torch.full((act_dim,), -0.7))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.logstd.clamp(-2.5, 0.0).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def env_action(u, a, arm_step):
    """정책의 행동(-1~1)을 환경의 행동(관절 목표각 6개 + 그리퍼 목표값)으로 바꾼다."""
    q = u.agent.robot.get_qpos()[:, :6]
    grip = torch.where(a[:, 6:7] > 0, GRIPPER_CLOSED, 0.0)
    return torch.cat([q + arm_step * a[:, :6], grip], 1)


class DemoBank:
    """시연 상태 묶음. 끝에서 offset 스텝 전의 상태를 꺼내 준다."""

    def __init__(self, path, device, obj_actor):
        b = np.load(path)
        start, length = b["start"], b["length"]
        z = b[f"states/actors/{obj_actor}"][:, 2]
        lifted = np.zeros(len(z), dtype=bool)  # 그 시점까지 물체를 들어 올린 적이 있는지
        for s, n in zip(start, length):
            lifted[s:s + n] = np.maximum.accumulate(z[s:s + n] - z[s]) > LIFT_HEIGHT
        self.start = torch.as_tensor(start, device=device)
        self.length = torch.as_tensor(length, device=device)
        self.goal = torch.as_tensor(b["goal_idx"], device=device)
        self.lifted = torch.as_tensor(lifted, device=device)
        self.states = {k[len("states/"):]: torch.as_tensor(b[k], device=device) for k in b.files if k.startswith("states/")}
        self.max_len = int(np.percentile(length, 95))  # 유난히 긴 시연 몇 개 때문에 진도가 늘어지지 않게 한다

    def sample(self, n, frontier, q=0.5, mix=0.0):
        ep = torch.randint(0, len(self.start), (n,), device=self.start.device)
        # 대부분 경계(frontier)에서 시작하고, 일부는 그보다 조금 뒤(끝 쪽)에서 시작한다
        back = torch.distributions.Geometric(torch.tensor(q, device=ep.device)).sample((n,)).long()
        # mix 비율만큼은 이미 지나온 구간 전체(0~frontier)에서 고르게 뽑는다. 경계만 연습하면 뒤 구간을 잊는다
        # (머스터드 병: 경계 174 에서 머무는 동안 끝에서 158 지점의 성공률이 0.85 → 0.36 으로 떨어졌다).
        spread = torch.rand(n, device=ep.device) < mix
        back = torch.where(spread, 1 + (torch.rand(n, device=ep.device) * frontier).long(), back)
        offset = torch.clamp(torch.clamp(frontier - back, min=0), max=self.length[ep] - 1)
        idx = self.start[ep] + self.length[ep] - 1 - offset
        sd = {"actors": {}, "articulations": {}, "goal_idx": self.goal[ep], "lifted": self.lifted[idx]}
        for key, arr in self.states.items():
            kind, name = key.split("/")
            sd[kind][name] = arr[idx]
        return sd, offset, back == 0


class Group:
    """물체 하나의 병렬 환경 묶음과 그 물체의 진도."""

    def __init__(self, key, index, num_objects, n, bank_path, seed, frontier, mix=0.0, gripper_speed=0.0):
        self.key, self.n, self.mix, self.forward_mix = key, n, mix, 0.0
        self.best_ep = 0.0
        self.env = gym.make(RL_ENV_IDS[key], num_envs=n, obs_mode="state", control_mode="pd_joint_pos",
                            sim_backend="gpu", reward_mode="sparse", gripper_speed=gripper_speed)
        self.u = self.env.unwrapped
        self.obs, _ = self.env.reset(seed=seed)
        dev = self.obs.device
        self.onehot = torch.zeros((n, num_objects), device=dev)
        self.onehot[:, index] = 1.0
        self.bank = DemoBank(bank_path, dev, OBJECT_ACTOR.get(key, "object"))
        self.frontier, self.forward = frontier, False
        self.budget = torch.zeros(n, device=dev, dtype=torch.long)       # 이번 회차에 허용된 스텝 수
        self.ep_len = torch.zeros(n, device=dev, dtype=torch.long)
        self.at_frontier = torch.zeros(n, device=dev, dtype=torch.bool)  # 경계에서 시작한 회차인지
        self.win_done = self.win_succ = self.ep_done = self.ep_succ = 0
        self.tog = self.da2 = self.steps = 0.0  # 기록용: 그리퍼 명령이 바뀐 횟수, 팔 행동 변화량 제곱합, 스텝 수
        self.win_rate, self.eval, self.eval_len = 0.0, None, None

    def full(self, env_obs, prev_a):
        return torch.cat([env_obs, self.onehot, prev_a], 1)

    def reset(self, idx):
        """idx 의 환경들을 다시 시작시킨다. 뒤에서부터 배우는 단계면 시연 상태로, 아니면 과제의 무작위 시작으로."""
        if self.forward:
            o, _ = self.env.reset(options={"env_idx": idx})
            self.budget[idx], self.at_frontier[idx] = 400, True
            # 무작위 시작 단계에서도 일부(forward_mix)는 시연의 임의 지점에서 시작해, 뒤쪽 구간(옮기기·놓기) 연습을 유지한다
            sub = idx[torch.rand(len(idx), device=idx.device) < self.forward_mix]
            if len(sub):
                sd, offset, _ = self.bank.sample(len(sub), self.bank.max_len, mix=1.0)
                o, _ = self.env.reset(options={"env_idx": sub, "reset_to_env_states": {"env_states": sd}})
                self.budget[sub], self.at_frontier[sub] = torch.clamp(offset * 2 + 40, max=400), False
        else:
            sd, offset, exact = self.bank.sample(len(idx), self.frontier, mix=self.mix)
            o, _ = self.env.reset(options={"env_idx": idx, "reset_to_env_states": {"env_states": sd}})
            self.budget[idx] = torch.clamp(offset * 2 + 40, max=400)
            self.at_frontier[idx] = exact
        self.ep_len[idx] = 0
        return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objects", default="cube,banana,mustard,baseball")
    p.add_argument("--bank-dir", default="../data/banks", help="bank_<물체>.npz 가 있는 폴더")
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=2048, help="물체 하나당 병렬 환경 수")
    p.add_argument("--env-counts", default="", help="물체별 환경 수를 따로 정할 때. 예: mustard=5120,cube=1024 (없는 물체는 --num-envs)")
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--max-iters", type=int, default=10**9)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--desired-kl", type=float, default=0.01)
    p.add_argument("--arm-step", type=float, default=0.03, help="행동 1 이 뜻하는 관절각 증분(rad). 0.03 이면 최대 0.6 rad/s")
    p.add_argument("--rate-coef", type=float, default=0.002, help="행동 변화량 제곱 평균에 곱하는 벌점 계수")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="팔 행동 저역 통과(0~1): 실행 행동 = (1-s)*정책 출력 + s*직전 실행 행동. 관측의 '직전 행동'과 벌점도 실행 행동 기준")
    p.add_argument("--gripper-speed", type=float, default=0.0, help="그리퍼가 닫히는 최대 속도(m/s). 0 이면 제한 없음")
    p.add_argument("--gripper-speed-start", type=float, default=0.0,
                   help="0 보다 크면 이 속도에서 시작해 성공률이 유지되는 동안 --gripper-speed 까지 단계적으로 늦춘다")
    p.add_argument("--gripper-anneal", type=float, default=0.9, help="한 번 늦출 때 속도에 곱하는 값")
    p.add_argument("--gripper-gate", type=float, default=0.9, help="모든 물체의 회차 성공률이 지금까지 최고치의 이 비율 이상일 때만 늦춘다")
    p.add_argument("--grip-coef", type=float, default=0.0, help="그리퍼 명령(열기/닫기)이 바뀔 때마다 주는 벌점")
    p.add_argument("--frontier-start", type=int, default=30)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--mix", type=float, default=0.3, help="뒤에서부터 배우는 단계에서, 지나온 구간 전체에서 고르게 시작하는 회차의 비율")
    p.add_argument("--advance-rate", type=float, default=0.85)
    p.add_argument("--advance-window", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--forward-mix", type=float, default=0.0, help="무작위 시작 단계에서 시연의 임의 지점에서 시작하는 회차의 비율")
    p.add_argument("--force-forward", default="", help="진도와 상관없이 지금부터 무작위 시작으로 학습할 물체(쉼표로 구분)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = args.objects.split(",")
    counts = {k: args.num_envs for k in keys}
    counts.update({kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.env_counts.split(",") if kv})
    grip_speed = args.gripper_speed_start if args.gripper_speed_start > 0 else args.gripper_speed
    if args.resume and args.gripper_speed > 0:  # 늦추던 도중에 끊겼으면 그 속도에서 잇는다
        saved = torch.load(args.resume, map_location="cpu").get("gripper_speed", 0.0)
        if saved > 0:
            grip_speed = max(args.gripper_speed, min(grip_speed, saved))
    groups = [Group(k, i, len(keys), counts[k], f"{args.bank_dir}/bank_{k}.npz", args.seed + i, args.frontier_start, args.mix,
                    gripper_speed=grip_speed)
              for i, k in enumerate(keys)]
    G, T = len(keys), args.num_steps
    N = sum(g.n for g in groups)
    dev = groups[0].obs.device
    act_dim = 7
    env_obs_dim = groups[0].obs.shape[1]
    obs_dim = env_obs_dim + G + act_dim
    hidden = (512, 512, 256)
    agent = Agent(obs_dim, act_dim, hidden).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    norm = RunningNorm(obs_dim, dev)
    it0, env_steps, lr = 0, 0, 3e-4
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        agent.load_state_dict(ck["agent"])
        norm.mean, norm.var, norm.count = ck["norm_mean"], ck["norm_var"], ck["norm_count"]
        it0, env_steps, lr = ck["iter"], ck["env_steps"], ck.get("lr", 1e-4)
        for g in groups:
            g.frontier, g.forward = ck["frontier"][g.key], ck["forward"][g.key]
    for g in groups:
        g.forward_mix = args.forward_mix
        if g.key in args.force_forward.split(","):
            g.forward = True

    prev_a = torch.zeros((N, act_dim), device=dev)
    edges = np.cumsum([0] + [g.n for g in groups])
    sl = [slice(int(edges[i]), int(edges[i + 1])) for i in range(G)]

    def reset_all():
        for g in groups:
            g.obs = g.reset(torch.arange(g.n, device=dev))
        prev_a.zero_()

    def evaluate(g, steps=400):
        """과제의 원래 무작위 시작에서, 잡음 없이 실행해 첫 회차의 성공률과 성공까지 걸린 스텝 수를 잰다."""
        n = g.n
        o, _ = g.env.reset()
        prev = torch.zeros((n, act_dim), device=dev)
        finished = torch.zeros(n, device=dev, dtype=torch.bool)
        succ = torch.zeros(n, device=dev, dtype=torch.bool)
        length = torch.zeros(n, device=dev)
        tog, da2, cnt = torch.zeros(n, device=dev), 0.0, 0.0
        with torch.no_grad():
            for t in range(steps):
                a = agent.actor(norm(g.full(o, prev))).clamp(-1, 1)
                if args.smooth > 0:
                    a[:, :6] = (1 - args.smooth) * a[:, :6] + args.smooth * prev[:, :6]
                live = ~finished
                tog += (((a[:, 6] > 0) != (prev[:, 6] > 0)) & live).float()
                if t > 0:
                    da2, cnt = da2 + (a[:, :6] - prev[:, :6]).pow(2).mean(-1)[live].sum().item(), cnt + live.sum().item()
                o, _, term, _, info = g.env.step(env_action(g.u, a, args.arm_step))
                prev = a.clone()
                new = term & ~finished
                succ |= new & info["success"]
                length[new] = t + 1
                finished |= term
                idx = torch.nonzero(term).squeeze(-1)
                if len(idx):
                    o, _ = g.env.reset(options={"env_idx": idx})
                    prev[idx] = 0
        # 잡음 없는 실행에서의 매끄러움: 성공 회차의 그리퍼 명령 바뀜 횟수(2 가 이상적), 팔 행동 변화량의 RMS
        g.eval_tog = tog[succ].mean().item() if succ.any() else float("nan")
        g.eval_da = (da2 / max(cnt, 1)) ** 0.5
        return succ.float().mean().item(), (length[succ].mean().item() if succ.any() else float("nan"))

    reset_all()
    log = open(out / "train.log", "a")
    b_obs = torch.zeros((T, N, obs_dim), device=dev)
    b_act = torch.zeros((T, N, act_dim), device=dev)
    b_logp, b_rew, b_done, b_val = (torch.zeros((T, N), device=dev) for _ in range(4))
    t_start = time.time()

    for it in range(it0 + 1, args.max_iters + 1):
        t_it = time.time()
        with torch.no_grad():
            for t in range(T):
                full = torch.cat([g.full(g.obs, prev_a[s]) for g, s in zip(groups, sl)])
                norm.update(full)
                nobs = norm(full)
                d = agent.dist(nobs)
                act = d.sample()
                b_obs[t], b_act[t], b_logp[t], b_val[t] = nobs, act, d.log_prob(act).sum(-1), agent.value(nobs)
                a = act.clamp(-1, 1)
                if args.smooth > 0:
                    a[:, :6] = (1 - args.smooth) * a[:, :6] + args.smooth * prev_a[:, :6]
                da2 = (a[:, :6] - prev_a[:, :6]).pow(2).mean(-1)
                toggle = ((a[:, 6] > 0) != (prev_a[:, 6] > 0)).float()
                pen = args.rate_coef * da2 + args.grip_coef * toggle
                prev_a = a.clone()
                for g, s in zip(groups, sl):
                    g.tog, g.da2, g.steps = g.tog + toggle[s].sum().item(), g.da2 + da2[s].sum().item(), g.steps + g.n
                    nxt, _, term, _, info = g.env.step(env_action(g.u, a[s], args.arm_step))
                    g.ep_len += 1
                    success = info["success"]
                    rew = success.float() - pen[s]
                    trunc = (g.ep_len >= g.budget) & ~term
                    if trunc.any():  # 시간 초과로 끊긴 회차는 그 뒤의 가치를 더해 준다
                        rew[trunc] += args.gamma * agent.value(norm(g.full(nxt, a[s])[trunc]))
                    done = term | trunc
                    b_rew[t, s], b_done[t, s] = rew, done.float()
                    idx = torch.nonzero(done).squeeze(-1)
                    if len(idx):
                        fr = g.at_frontier[idx]
                        g.ep_done += len(idx)
                        g.ep_succ += int(success[idx].sum())
                        g.win_done += int(fr.sum())
                        g.win_succ += int((success[idx] & fr).sum())
                        nxt = g.reset(idx)
                        prev_a[s][idx] = 0
                    g.obs = nxt
            last_val = agent.value(norm(torch.cat([g.full(g.obs, prev_a[s]) for g, s in zip(groups, sl)])))
            adv = torch.zeros_like(b_rew)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else b_val[t + 1]
                nonterminal = 1.0 - b_done[t]
                delta = b_rew[t] + args.gamma * next_val * nonterminal - b_val[t]
                gae = delta + args.gamma * args.lam * nonterminal * gae
                adv[t] = gae
            ret = adv + b_val
        env_steps += N * T

        f_obs, f_act, f_logp = b_obs.reshape(-1, obs_dim), b_act.reshape(-1, act_dim), b_logp.reshape(-1)
        f_adv, f_ret = adv.reshape(-1), ret.reshape(-1)
        mb = N * T // args.minibatches
        kls = []
        for _ in range(args.epochs):
            perm = torch.randperm(N * T, device=dev)
            for k in range(args.minibatches):
                j = perm[k * mb:(k + 1) * mb]
                d = agent.dist(f_obs[j])
                logp = d.log_prob(f_act[j]).sum(-1)
                ratio = (logp - f_logp[j]).exp()
                a_ = (f_adv[j] - f_adv[j].mean()) / (f_adv[j].std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(0.8, 1.2) * a_).mean()
                v_loss = 0.5 * (agent.value(f_obs[j]) - f_ret[j]).pow(2).mean()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - f_logp[j])).mean().item()
                kls.append(kl)
                if kl > args.desired_kl * 2:  # 정책이 한 번에 너무 많이 바뀌면 학습률을 낮춘다
                    lr = max(1e-5, lr / 1.5)
                elif kl < args.desired_kl / 2:
                    lr = min(1e-3, lr * 1.5)
                for pg_ in opt.param_groups:
                    pg_["lr"] = lr
                opt.zero_grad()
                (pg + v_loss).backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

        # 물체마다, 경계에서 시작한 회차의 성공률로 진도를 정한다
        note = ""
        for g in groups:
            if g.win_done >= args.advance_window:
                g.win_rate = g.win_succ / g.win_done
                if g.win_rate >= args.advance_rate and not g.forward:
                    if g.frontier >= g.bank.max_len:
                        g.forward = True
                        note += f" [{g.key} -> 무작위 시작]"
                    else:
                        g.frontier = min(g.bank.max_len, g.frontier + args.stride)
                g.win_done = g.win_succ = 0

        if it % args.eval_every == 0:
            for g in groups:
                g.eval, g.eval_len = evaluate(g)
            reset_all()
        if it % 10 == 0 or note:
            parts = []
            for g in groups:
                where = "무작위" if g.forward else f"{g.frontier}/{g.bank.max_len}"
                ev = "-" if g.eval is None else f"{g.eval:.2f}({g.eval_len:.0f})"
                if g.eval is not None and hasattr(g, "eval_tog"):
                    ev += f" etg {g.eval_tog:.1f} eda {g.eval_da:.3f}"
                # tg: 100스텝당 그리퍼 명령이 바뀐 횟수, da: 팔 행동 변화량의 RMS (탐색 잡음 포함)
                g.last_ep = g.ep_succ / max(g.ep_done, 1)
                g.best_ep = max(g.best_ep, g.last_ep)
                parts.append(f"{g.key} {where} sr {g.win_rate:.2f} ep {g.ep_succ / max(g.ep_done, 1):.2f} eval {ev} "
                             f"tg {100 * g.tog / max(g.steps, 1):.1f} da {(g.da2 / max(g.steps, 1)) ** 0.5:.3f}")
                g.ep_done = g.ep_succ = 0
                g.tog = g.da2 = g.steps = 0.0
            if args.gripper_speed > 0:
                parts.append(f"grip {grip_speed * 1000:.0f}mm/s")
                # 성공률이 유지되면 그리퍼를 한 단계 더 느리게 한다 (첫 기록은 기준을 잡는 데만 쓴다)
                if (grip_speed > args.gripper_speed and it - it0 >= 20 and it % 10 == 0
                        and all(g.last_ep >= args.gripper_gate * g.best_ep for g in groups)):
                    grip_speed = max(args.gripper_speed, grip_speed * args.gripper_anneal)
                    for g in groups:
                        g.u.gripper_speed = grip_speed
            line = (f"iter {it} steps {env_steps / 1e6:.1f}M fps {N * T / (time.time() - t_it):.0f} | " + " | ".join(parts) +
                    f" | kl {np.mean(kls):.4f} lr {lr:.1e} std {agent.logstd.clamp(-2.5, 0).exp().mean().item():.2f} "
                    f"hours {(time.time() - t_start) / 3600:.2f}{note}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        stop = (time.time() - t_start) > args.hours * 3600 or it >= args.max_iters
        if it % 50 == 0 or stop:
            torch.save({"agent": agent.state_dict(), "norm_mean": norm.mean, "norm_var": norm.var, "norm_count": norm.count,
                        "frontier": {g.key: g.frontier for g in groups}, "forward": {g.key: g.forward for g in groups},
                        "eval": {g.key: g.eval for g in groups}, "iter": it, "env_steps": env_steps,
                        "objects": keys, "obs_dim": obs_dim, "env_obs_dim": env_obs_dim, "act_dim": act_dim, "hidden": hidden,
                        "arm_step": args.arm_step, "smooth": args.smooth, "gripper_speed": grip_speed if args.gripper_speed > 0 else 0.0,
                        "gripper_speed_target": args.gripper_speed, "lr": lr, "version": 2}, out / "latest.pt")
            (out / "status.json").write_text(json.dumps({
                "iter": it, "env_steps": env_steps, "hours": (time.time() - t_start) / 3600,
                "objects": {g.key: {"frontier": g.frontier, "max_len": g.bank.max_len, "forward": g.forward,
                                    "eval": g.eval, "eval_len": g.eval_len} for g in groups}}))
        if stop:
            break
    for g in groups:
        g.env.close()


if __name__ == "__main__":
    main()
