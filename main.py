"""
FAIR-QMIX-PDQN-MAML-CEM – CORRECTED IMPLEMENTATION
Optimized Hierarchical Joint Scheduling for User-Edge-Cloud Systems

Novel components (exactly matching algorithm spec):
  1. FAIR-QMIX  – QMIX value decomposition + fairness-aware reward shaping
  2. APSN        – Adaptive Parameter Space Noise with KL-based σ adaptation
  3. DPER        – Diversity-Prioritised Experience Replay with novelty detection
  4. MAML        – Model-Agnostic Meta-Learning warm-start for intra-slice DQN
  5. P-DQN       – Parametrised DQN for hybrid discrete-continuous actions
  6. CEM         – Cross-Entropy Method joint discrete-continuous optimisation
  7. Batched convex solver with warm-started dual variables

KEY BUG-FIXES vs. original:
  • Each agent stores / trains on its OWN state vector (state_dim=6), not a
    concatenated global state – eliminating the [64×0] × [6×128] matmul crash.
  • CriticNetwork input is (state_dim + action_dim), not (K*state + K*action).
  • QMIXHypernetwork state input is the GLOBAL state (K*state_dim), kept as
    a separate concatenation from the per-agent actor/critic inputs.
  • APSN KL update is actually applied after each slice scheduling step.
  • Buffer stores individual (s_k, a_k, r_k, s'_k) tuples, not a flat mix.
  • P-DQN feature/q/policy dimensions are consistent end-to-end.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Dict
import copy
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

# ─── reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

DEVICE = torch.device("cpu")


# ============================================================================
# 1.  NEURAL NETWORK ARCHITECTURES
# ============================================================================

class ActorNetwork(nn.Module):
    """Per-agent actor: maps individual agent observation → action."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentQNetwork(nn.Module):
    """Per-agent Q-network: maps (individual_state, individual_action) → scalar Q."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                  nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))   # (B, 1)


class QMIXHypernetwork(nn.Module):
    """
    QMIX mixing network (Algorithm 1, Lines 26-28).
    Combines per-agent Qs into Q_total with monotonicity constraint.
    Conditioning is on the GLOBAL state (concatenation of all agents' states).
    """
    def __init__(self, n_agents: int, global_state_dim: int, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.hidden   = hidden
        # Hyper-networks produce *non-negative* weights (torch.abs)
        self.hyper_w1 = nn.Linear(global_state_dim, hidden * n_agents)
        self.hyper_w2 = nn.Linear(global_state_dim, hidden)
        self.hyper_b1 = nn.Linear(global_state_dim, hidden)
        self.hyper_b2 = nn.Linear(global_state_dim, 1)

    def forward(self, agent_qs: torch.Tensor, global_states: torch.Tensor) -> torch.Tensor:
        """
        agent_qs:      (B, n_agents)
        global_states: (B, global_state_dim)
        returns:       (B, 1)
        """
        B = agent_qs.shape[0]
        # Layer 1  ─  (B, hidden, n_agents) × (B, n_agents, 1) → (B, hidden)
        w1 = torch.abs(self.hyper_w1(global_states)).view(B, self.hidden, self.n_agents)
        b1 = self.hyper_b1(global_states)                                # (B, hidden)
        hidden = torch.relu(
            torch.bmm(w1, agent_qs.unsqueeze(-1)).squeeze(-1) + b1
        )                                                                 # (B, hidden)
        # Layer 2  ─  (B, 1, hidden) × (B, hidden, 1) → (B, 1)
        w2 = torch.abs(self.hyper_w2(global_states)).unsqueeze(1)        # (B, 1, hidden)
        b2 = self.hyper_b2(global_states)                                # (B, 1)
        q_total = torch.bmm(w2, hidden.unsqueeze(-1)).squeeze(-1) + b2   # (B, 1)
        return q_total                                                    # (B, 1)


class AutoencoderNovelty(nn.Module):
    """Small autoencoder used by DPER for novelty detection."""
    def __init__(self, input_dim: int, latent: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128,  64),       nn.ReLU(),
            nn.Linear(64,  latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent,  64),    nn.ReLU(),
            nn.Linear(64,  128),       nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, self.decoder(z)


class P_DQN(nn.Module):
    """
    Parametrised DQN (Algorithm 2, Lines 16-20).

    Jointly learns:
      • Q(s, d, ρ)  – value for discrete action d given state s and ratio ρ
      • μ(s, d)     – deterministic continuous action ρ for each discrete d

    Architecture is shaped so ALL internal dimensions are consistent.
    """
    def __init__(self, state_dim: int, n_discrete: int, hidden: int = 256):
        super().__init__()
        self.n_discrete = n_discrete

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )                                                   # output: (B, hidden)

        # Policy head: (feature ∥ one-hot_d) → ρ ∈ [0,1]
        self.policy_net = nn.Sequential(
            nn.Linear(hidden + n_discrete, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

        # Q head: (feature ∥ ρ) → scalar Q
        self.q_net = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state: torch.Tensor):
        """
        state: (B, state_dim)
        returns:
            q_vals: (B, n_discrete)   – Q-value per discrete action
            rho:    (B, n_discrete)   – continuous ρ per discrete action
        """
        B    = state.shape[0]
        feat = self.feature(state)                                        # (B, H)

        # Expand feature for all discrete actions at once
        feat_exp = feat.unsqueeze(1).expand(-1, self.n_discrete, -1)     # (B, D, H)
        onehot   = torch.eye(self.n_discrete, device=state.device)       # (D, D)
        onehot   = onehot.unsqueeze(0).expand(B, -1, -1)                 # (B, D, D)

        # Policy: predict ρ for every discrete action simultaneously
        rho = self.policy_net(
            torch.cat([feat_exp, onehot], dim=-1)                        # (B, D, H+D)
        ).squeeze(-1)                                                     # (B, D)

        # Q-value: evaluate (feat, ρ) for every action
        q_vals = self.q_net(
            torch.cat([feat_exp, rho.unsqueeze(-1)], dim=-1)             # (B, D, H+1)
        ).squeeze(-1)                                                     # (B, D)

        return q_vals, rho


# ============================================================================
# 2.  DIVERSITY-PRIORITISED REPLAY BUFFER  (DPER)
#     Algorithm 1, Lines 20-23
# ============================================================================

class DPERBuffer:
    """
    Stores (s, a, r, s') tuples for ONE agent (not global).
    Priority = TD-error priority (PER) + novelty score (diversity).
    """
    def __init__(self, capacity: int = 200_000, alpha: float = 0.6,
                 beta: float = 0.4, novelty_coef: float = 0.5):
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta         = beta
        self.novelty_coef = novelty_coef
        self.buffer: list = []
        self.priorities   = np.zeros(capacity, dtype=np.float32)
        self.pos          = 0
        self.size         = 0
        self.ae: AutoencoderNovelty = None   # lazy-init on first add
        self.prototypes: list = []
        self._step        = 0

    # ── autoencoder helpers ──────────────────────────────────────────────────
    def _init_ae(self, dim: int):
        self.ae = AutoencoderNovelty(dim)
        self.ae.eval()

    def _encode(self, sa: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z, _ = self.ae(torch.FloatTensor(sa).unsqueeze(0))
        return z.numpy().flatten()

    def _novelty(self, sa: np.ndarray) -> float:
        if self.ae is None or len(self.prototypes) == 0:
            return 1.0
        z     = self._encode(sa)
        dists = [np.linalg.norm(z - p) for p in self.prototypes]
        return min(dists) / (1.0 + max(dists) + 1e-8)

    # ── public API ───────────────────────────────────────────────────────────
    def add(self, transition: tuple, td_error: float, state_action: np.ndarray):
        """transition = (s, a, r, s')"""
        if self.ae is None:
            self._init_ae(len(state_action))

        novelty  = self._novelty(state_action)
        priority = (abs(td_error) + 1e-6) ** self.alpha + self.novelty_coef * novelty

        if self.size < self.capacity:
            self.buffer.append(transition)
            self.priorities[self.size] = priority
            self.size += 1
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self._step += 1

        # Periodically refresh prototypes (Algorithm 1, Lines 42-43)
        if self._step % 200 == 0 and self.size >= 100:
            self._update_prototypes()

    def sample(self, batch_size: int):
        if self.size < batch_size:
            return None, None, None
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        idx     = np.random.choice(self.size, batch_size, replace=False, p=probs)
        batch   = [self.buffer[i] for i in idx]
        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()
        return batch, idx, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = (abs(td) + 1e-6) ** self.alpha

    def _update_prototypes(self):
        """k-means in latent space (Algorithm 1, Lines 42-43)."""
        n_sample = min(self.size, 500)
        indices  = np.random.choice(self.size, n_sample, replace=False)
        latents  = []
        for i in indices:
            s, a = self.buffer[i][0], self.buffer[i][1]
            sa   = np.concatenate([np.asarray(s).flatten(), np.asarray(a).flatten()])
            latents.append(self._encode(sa))
        if len(latents) >= 20:
            n_clust = min(10, len(latents) // 10)
            if n_clust >= 2:
                km = MiniBatchKMeans(n_clusters=n_clust, random_state=42)
                km.fit(np.array(latents))
                self.prototypes = km.cluster_centers_.tolist()


# ============================================================================
# 3.  FAIR-QMIX-APSN AGENT  (Inter-slice resource allocation)
#     Algorithm 1 (complete)
# ============================================================================

class FAIR_QMIX_APSN_Agent:
    """
    Implements Algorithm 1: FAIR-QMIX-APSN for Inter-Slice Resource Allocation.

    Design decisions that avoid the shape crash:
      • Each actor    takes (state_dim,)          as input.
      • Each Q-net   takes (state_dim + action_dim) as input (per-agent).
      • QMIX         takes per-agent Q scalars  + GLOBAL state (K*state_dim).
      • Replay buffer stores individual (s_k, a_k, r_k, s'_k) tuples.
      • train_step() rebuilds global state by concatenating per-agent states.
    """

    def __init__(self, n_slices: int, state_dim: int, action_dim: int,
                 fairness_coef: float = 0.1, fairness_std: float = 0.05):
        self.n_slices     = n_slices
        self.state_dim    = state_dim       # per-agent observation size (= 6)
        self.action_dim   = action_dim      # per-agent action size      (= 3)
        self.global_dim   = state_dim * n_slices   # for QMIX conditioning
        self.fairness_coef = fairness_coef
        self.fairness_std  = fairness_std

        # ── Actor networks (one per slice) ───────────────────────────────────
        self.actors        = [ActorNetwork(state_dim, action_dim).to(DEVICE)
                              for _ in range(n_slices)]
        self.target_actors = [copy.deepcopy(a) for a in self.actors]

        # ── Per-agent Q-networks ─────────────────────────────────────────────
        # Input: (state_dim + action_dim), output: scalar
        self.agent_qs        = [AgentQNetwork(state_dim, action_dim).to(DEVICE)
                                for _ in range(n_slices)]
        self.target_agent_qs = [copy.deepcopy(q) for q in self.agent_qs]

        # ── QMIX mixing network ──────────────────────────────────────────────
        # Conditioned on global state = concatenation of all per-agent states
        self.qmix        = QMIXHypernetwork(n_slices, self.global_dim).to(DEVICE)
        self.target_qmix = copy.deepcopy(self.qmix)

        # ── Optimisers ───────────────────────────────────────────────────────
        self.actor_opts   = [optim.Adam(a.parameters(), lr=1e-4)
                             for a in self.actors]
        self.q_opts       = [optim.Adam(q.parameters(), lr=1e-3)
                             for q in self.agent_qs]
        self.qmix_opt     = optim.Adam(self.qmix.parameters(), lr=1e-3)

        # ── APSN (Algorithm 1, Lines 12-16, 34-37) ──────────────────────────
        self.noise_scales = [0.1] * n_slices   # σ_k per agent
        self.target_kl    = 0.01               # d_target
        self.eta_sigma    = 0.01               # η_σ
        self.sigma_min    = 1e-4
        self.sigma_max    = 1.0

        # ── Replay buffers (one per agent for clean per-agent tuples) ────────
        self.replays = [DPERBuffer(capacity=50_000) for _ in range(n_slices)]

        # ── Hyper-params ─────────────────────────────────────────────────────
        self.gamma      = 0.95
        self.tau        = 0.01
        self.batch_size = 64

        # ── Running state for global-state assembly during training ──────────
        # We keep the last-seen per-agent states so we can build the global vec
        self._last_states      = [np.zeros(state_dim) for _ in range(n_slices)]
        self._last_next_states = [np.zeros(state_dim) for _ in range(n_slices)]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _to_t(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(arr).to(DEVICE)

    def _soft_update(self):
        """Algorithm 1, Lines 38-41."""
        for k in range(self.n_slices):
            for p, tp in zip(self.actors[k].parameters(),
                             self.target_actors[k].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.agent_qs[k].parameters(),
                             self.target_agent_qs[k].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.qmix.parameters(), self.target_qmix.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _compute_kl(self, k: int) -> float:
        """
        Approximate KL divergence between π_k(·;θ_k) and π_k(·;θ̃_k)
        using Monte-Carlo samples in action space (Algorithm 1, Line 35).
        """
        s = self._to_t(self._last_states[k]).unsqueeze(0)   # (1, state_dim)
        with torch.no_grad():
            a_clean = self.actors[k](s).squeeze(0)

            # Perturb and evaluate
            perturbed = copy.deepcopy(self.actors[k])
            with torch.no_grad():
                for p in perturbed.parameters():
                    p.data += torch.randn_like(p) * self.noise_scales[k]
            a_noisy = perturbed(s).squeeze(0)

        # KL for Gaussian approximation:  0.5 * ||μ1-μ2||² / σ² (σ=0.1)
        diff  = (a_clean - a_noisy).pow(2).mean().item()
        return 0.5 * diff / (0.1 ** 2)

    # ── public API ───────────────────────────────────────────────────────────

    def fairness_reward(self, profits: List[float]) -> List[float]:
        """
        Algorithm 1, Line 18:
          r_k = profit_k + λ·log(profit_k + ε) − μ·std(profits)
        """
        std_p = float(np.std(profits))
        return [
            p + self.fairness_coef * np.log(max(p, 1e-6)) - self.fairness_std * std_p
            for p in profits
        ]

    def select_action(self, states: List[np.ndarray], explore: bool = True) -> List[np.ndarray]:
        """
        Algorithm 1, Lines 12-16:  APSN exploration (perturb parameters, not actions).
        """
        actions = []
        for k, state in enumerate(states):
            self._last_states[k] = state.copy()
            s_t = self._to_t(state).unsqueeze(0)    # (1, state_dim) – correct shape

            if explore:
                # Save original params, add Gaussian noise, evaluate, restore
                orig = [p.clone() for p in self.actors[k].parameters()]
                with torch.no_grad():
                    for p in self.actors[k].parameters():
                        p.data += torch.randn_like(p) * self.noise_scales[k]
                    a = self.actors[k](s_t).cpu().numpy()[0]
                for p, o in zip(self.actors[k].parameters(), orig):
                    p.data = o
            else:
                with torch.no_grad():
                    a = self.actors[k](s_t).cpu().numpy()[0]

            actions.append(np.clip(a, 0.05, 0.95))
        return actions

    def store_transition(self, k: int, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray):
        """Algorithm 1, Lines 20-23: store with DPER priority."""
        self._last_next_states[k] = next_state.copy()
        td    = abs(reward)
        sa    = np.concatenate([state.flatten(), action.flatten()])
        self.replays[k].add((state, action, reward, next_state), td, sa)

    def _update_apsn(self):
        """Algorithm 1, Lines 34-37: adapt per-agent noise scales."""
        for k in range(self.n_slices):
            d_kl = self._compute_kl(k)
            self.noise_scales[k] *= np.exp(self.eta_sigma * (d_kl - self.target_kl))
            self.noise_scales[k]  = float(np.clip(self.noise_scales[k],
                                                   self.sigma_min, self.sigma_max))

    def train_step(self) -> Dict:
        """Full Algorithm 1 training update for one batch."""
        # Need at least batch_size samples in ALL agents' buffers
        if any(r.size < self.batch_size for r in self.replays):
            return {}

        # ── Sample per-agent batches ─────────────────────────────────────────
        batches, all_idx, all_w = [], [], []
        for k in range(self.n_slices):
            b, idx, w = self.replays[k].sample(self.batch_size)
            if b is None:
                return {}
            batches.append(b)
            all_idx.append(idx)
            all_w.append(w)

        # ── Unpack per-agent tensors ─────────────────────────────────────────
        # Each:  states_k  → (B, state_dim)
        #        actions_k → (B, action_dim)
        #        rewards_k → (B,)
        #        nstates_k → (B, state_dim)
        states_list  = []
        actions_list = []
        rewards_list = []
        nstates_list = []
        for k in range(self.n_slices):
            b = batches[k]
            states_list.append(
                torch.FloatTensor(np.stack([t[0].flatten() for t in b])).to(DEVICE))
            actions_list.append(
                torch.FloatTensor(np.stack([np.asarray(t[1]).flatten() for t in b])).to(DEVICE))
            rewards_list.append(
                torch.FloatTensor([t[2] for t in b]).to(DEVICE))
            nstates_list.append(
                torch.FloatTensor(np.stack([t[3].flatten() for t in b])).to(DEVICE))

        # ── Build global state tensors for QMIX ─────────────────────────────
        # global_states:  (B, K*state_dim)
        global_states  = torch.cat(states_list,  dim=1)   # (B, global_dim)
        global_nstates = torch.cat(nstates_list, dim=1)   # (B, global_dim)

        # Mean reward across slices for shared QMIX target
        rewards_mean = torch.stack(rewards_list, dim=1).mean(dim=1)  # (B,)

        # ── Compute QMIX target (Algorithm 1, Lines 24-29) ──────────────────
        with torch.no_grad():
            # Target actors produce next actions for each agent
            next_actions = [
                self.target_actors[k](nstates_list[k])   # (B, action_dim)
                for k in range(self.n_slices)
            ]
            # Target per-agent Q values → (B, 1) each → stack → (B, K)
            target_qs = torch.cat(
                [self.target_agent_qs[k](nstates_list[k], next_actions[k])
                 for k in range(self.n_slices)],
                dim=1
            )                                             # (B, K)
            # QMIX mixing on global next state
            q_tot_next  = self.target_qmix(target_qs, global_nstates).squeeze(-1)  # (B,)
            q_tot_target = rewards_mean + self.gamma * q_tot_next                  # (B,)

        # ── Current per-agent Qs → mix ───────────────────────────────────────
        current_qs = torch.cat(
            [self.agent_qs[k](states_list[k], actions_list[k])
             for k in range(self.n_slices)],
            dim=1
        )                                                 # (B, K)
        q_tot_current = self.qmix(current_qs, global_states).squeeze(-1)  # (B,)

        # ── Critic / QMIX loss (Algorithm 1, Line 29-30) ────────────────────
        weights_t  = torch.FloatTensor(all_w[0]).to(DEVICE)  # use agent-0 IS weights
        td_errors  = (q_tot_target - q_tot_current).detach().cpu().numpy()
        critic_loss = (weights_t * (q_tot_target - q_tot_current).pow(2)).mean()

        for opt in self.q_opts:
            opt.zero_grad()
        self.qmix_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for q in self.agent_qs for p in q.parameters()], 10.0)
        for opt in self.q_opts:
            opt.step()
        self.qmix_opt.step()

        # Update DPER priorities
        for k in range(self.n_slices):
            self.replays[k].update_priorities(all_idx[k], td_errors)

        # ── Actor updates (Algorithm 1, Lines 31-33) ─────────────────────────
        actor_losses = []
        for k in range(self.n_slices):
            self.actor_opts[k].zero_grad()
            new_a_k = self.actors[k](states_list[k])    # (B, action_dim)

            # Temporarily substitute new action for agent k; keep others fixed
            new_actions = [
                (new_a_k if j == k else actions_list[j].detach())
                for j in range(self.n_slices)
            ]
            new_qs = torch.cat(
                [self.agent_qs[j](states_list[j], new_actions[j])
                 for j in range(self.n_slices)],
                dim=1
            )                                            # (B, K)
            total_q   = self.qmix(new_qs, global_states).mean()
            actor_loss = -total_q
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[k].parameters(), 10.0)
            self.actor_opts[k].step()
            actor_losses.append(actor_loss.item())

        # ── Soft target updates (Algorithm 1, Lines 38-41) ───────────────────
        self._soft_update()

        # ── APSN noise-scale update (Algorithm 1, Lines 34-37) ───────────────
        self._update_apsn()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss':  float(np.mean(actor_losses))
        }


# ============================================================================
# 4.  BATCHED CONVEX SOLVER WITH WARM-STARTING
#     Algorithm 2 – BATCHED_CONVEX_SOLVER function
# ============================================================================

class BatchedConvexSolver:
    """
    Warm-started solver for convex resource sub-problems P5-P7.
    Dual variables (μ, ν1, ν2) are cached per (task-type, offloading-mode) key.
    """
    def __init__(self):
        self._cache: Dict = {}      # key → (mu, nu1, nu2)

    def solve(self, rho: float, Y: int, task: dict, slice_res: dict) -> dict:
        data     = task['task_size_KB'] * 1000          # bits (proxy)
        cpu      = task['cpu_cycles_per_KB'] * data / 1000
        deadline = task['deadline_ms'] / 1000           # seconds

        B_max = slice_res.get('bandwidth_mhz', 100.0)
        C_max = slice_res.get('edge_cpu_ghz' if Y else 'cloud_cpu_ghz', 10.0)

        # Cache key groups tasks by type + offloading mode
        cache_key = (Y, int(data // 1000), int(cpu // 1e6))
        mu, nu1, nu2 = self._cache.get(cache_key, (1.0, 1.0, 1.0))

        # Closed-form KKT solutions (Equations 48, 56, 57 from paper)
        #   b_up*  ∝  √(ρ·data·cpu) / deadline          (Eq. 48 vectorised)
        #   f_m*   ∝  (ρ·cpu / deadline)^(1/3) / ν1     (Eq. 56)
        #   f_cl*  ∝  (ρ·cpu / deadline)^(1/3) / ν2     (Eq. 57)
        safe_dl   = deadline + 1e-9
        bw_opt    = np.sqrt(abs(rho * data * cpu) + 1e-9) / (mu * safe_dl * 10.0)
        fm_opt    = ((rho * cpu / safe_dl) ** (1/3)) / (nu1 * 1e3 + 1e-9) / 1e6
        fcl_opt   = ((rho * cpu / safe_dl) ** (1/3)) / (nu2 * 1e3 + 1e-9) / 1e6

        # Project to feasible range
        bw  = float(np.clip(bw_opt,  0.1, B_max))
        comp = float(np.clip(fm_opt if Y else fcl_opt, 0.1, C_max))

        # 2-3 Newton iterations to update dual variables (warm-start refinement)
        for _ in range(3):
            bw_used  = bw
            comp_used = comp
            mu  = max(1e-6, mu  + 0.1 * (bw_used  - B_max))
            nu1 = max(1e-6, nu1 + 0.1 * (comp_used - C_max)) if Y  else nu1
            nu2 = max(1e-6, nu2 + 0.1 * (comp_used - C_max)) if not Y else nu2

        self._cache[cache_key] = (mu, nu1, nu2)
        return {'bandwidth_mhz': bw, 'comp_ghz': comp, 'Y': Y, 'rho': rho}


# ============================================================================
# 5.  MAML-PDQN-CEM AGENT  (Intra-slice task offloading)
#     Algorithm 2 (complete)
# ============================================================================

class MAML_PDQN_CEM_Agent:
    """
    Implements Algorithm 2: MAML-PDQN-CEM for Intra-Slice Task Offloading.
    """
    def __init__(self, state_dim: int, n_discrete: int, n_edge: int,
                 meta_lr: float = 1e-3):
        self.state_dim  = state_dim
        self.n_discrete = n_discrete   # 2 * n_edge
        self.n_edge     = n_edge

        self.pdqn        = P_DQN(state_dim, n_discrete).to(DEVICE)
        self.target_pdqn = copy.deepcopy(self.pdqn)

        # Two optimisers: meta (for MAML outer loop) and online (for fast adapt)
        self.meta_opt   = optim.Adam(self.pdqn.parameters(), lr=meta_lr)
        self.online_opt = optim.Adam(self.pdqn.parameters(), lr=1e-4)

        self.replay = deque(maxlen=50_000)

        # CEM hyper-params (Algorithm 2, Lines 24-45)
        self.n_cem    = 5
        self.n_cand   = 30
        self.n_elite  = 8

        self.gamma        = 0.95
        self.batch_size   = 64
        self.target_update = 100
        self._step        = 0

        self.convex = BatchedConvexSolver()

    # ── loss helper ──────────────────────────────────────────────────────────

    def _loss(self, batch: list, model: P_DQN) -> torch.Tensor:
        """Shared DQN+policy loss for MAML inner loop and online training."""
        states     = torch.FloatTensor(np.stack([b[0] for b in batch])).to(DEVICE)
        acts       = torch.LongTensor([b[1] for b in batch]).to(DEVICE)
        rhos       = torch.FloatTensor([b[2] for b in batch]).to(DEVICE)
        rewards    = torch.FloatTensor([b[3] for b in batch]).to(DEVICE)
        nxt_states = torch.FloatTensor(np.stack([b[4] for b in batch])).to(DEVICE)
        dones      = torch.FloatTensor([b[5] for b in batch]).to(DEVICE)

        q_vals, pred_rho = model(states)                         # (B, D), (B, D)
        curr_q = q_vals.gather(1, acts.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            next_q, _ = self.target_pdqn(nxt_states)
            target_q  = rewards + self.gamma * (1 - dones) * next_q.max(dim=1)[0]

        q_loss  = nn.MSELoss()(curr_q, target_q)
        sel_rho = pred_rho.gather(1, acts.unsqueeze(1)).squeeze(1)
        rho_loss = nn.MSELoss()(sel_rho, rhos)

        return q_loss + 0.1 * rho_loss

    # ── MAML meta-training (Algorithm 2, Pre-training Phase) ─────────────────

    def maml_meta_train(self, task_list: list, inner_steps: int = 5,
                         outer_steps: int = 3):
        """
        Offline MAML pre-training (Algorithm 2, Steps 1-3).
        task_list: list of dicts with 'transitions' and 'val_transitions'.
        """
        print("  [MAML] meta-training…")
        for _ in range(outer_steps):
            meta_grads = []
            for task in task_list:
                if len(task['transitions']) < self.batch_size:
                    continue
                fast = copy.deepcopy(self.pdqn)
                fast_opt = optim.SGD(fast.parameters(),
                                     lr=self.meta_opt.param_groups[0]['lr'])
                for _ in range(inner_steps):
                    b = random.sample(task['transitions'],
                                      min(self.batch_size, len(task['transitions'])))
                    loss = self._loss(b, fast)
                    fast_opt.zero_grad()
                    loss.backward()
                    fast_opt.step()
                # Meta-gradient from validation set
                vb = random.sample(task['val_transitions'],
                                   min(self.batch_size, len(task['val_transitions'])))
                vl = self._loss(vb, fast)
                grads = torch.autograd.grad(vl, self.pdqn.parameters(),
                                            allow_unused=True)
                meta_grads.append([
                    g.detach().clone() if g is not None else torch.zeros_like(p)
                    for g, p in zip(grads, self.pdqn.parameters())
                ])
            if meta_grads:
                avg_g = [torch.stack([g[i] for g in meta_grads]).mean(0)
                         for i in range(len(meta_grads[0]))]
                self.meta_opt.zero_grad()
                for p, g in zip(self.pdqn.parameters(), avg_g):
                    p.grad = g
                self.meta_opt.step()
        print("  [MAML] done.")

    def rapid_adapt(self, recent_batch: list, n_steps: int = 10):
        """
        Algorithm 2, Steps 5-10: 5-10 gradient steps with recent data.
        Called at the start of each new slice scheduling slot T.
        """
        if len(recent_batch) < self.batch_size:
            return
        for _ in range(n_steps):
            b = random.sample(recent_batch, min(self.batch_size, len(recent_batch)))
            loss = self._loss(b, self.pdqn)
            self.online_opt.zero_grad()
            loss.backward()
            self.online_opt.step()

    # ── CEM action selection (Algorithm 2, Lines 14-47) ──────────────────────

    def select_action_with_cem(self, state: np.ndarray, slice_res: dict,
                                task: dict):
        """
        P-DQN initialisation followed by CEM joint optimisation.
        Returns: (best_discrete_action, best_rho, best_resources_dict)
        """
        s_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            qv, rh = self.pdqn(s_t)
            qv = qv[0].cpu().numpy()   # (n_discrete,)
            rh = rh[0].cpu().numpy()   # (n_discrete,)

        # Initialise CEM distributions from P-DQN output (Lines 21-22)
        exp_q  = np.exp(qv - qv.max())
        probs  = exp_q / (exp_q.sum() + 1e-8)
        d_star = int(np.argmax(qv))
        rho_mu = float(rh[d_star])
        rho_sd = 0.2

        best_r, best_d, best_rho, best_res = -np.inf, d_star, rho_mu, {}

        for _ in range(self.n_cem):           # CEM iterations (Line 24)
            # Sample N candidates (Line 26)
            candidates = [
                (int(np.random.choice(self.n_discrete, p=probs)),
                 float(np.clip(np.random.normal(rho_mu, rho_sd), 0.0, 1.0)))
                for _ in range(self.n_cand)
            ]

            # Evaluate each candidate (Lines 29-33)
            rew_list, res_list = [], []
            for d, rho in candidates:
                Y   = 1 if d >= self.n_edge else 0
                res = self.convex.solve(rho, Y, task, slice_res)
                rew = self._eval_candidate(rho, Y, res, task)
                rew_list.append(rew)
                res_list.append(res)

            # Select elites (Lines 36-41)
            elite_idx = np.argsort(rew_list)[-self.n_elite:]
            if rew_list[elite_idx[-1]] > best_r:
                best_r   = rew_list[elite_idx[-1]]
                best_d, best_rho = candidates[elite_idx[-1]]
                best_res = res_list[elite_idx[-1]]

            elite_rhos = [candidates[i][1] for i in elite_idx]
            rho_mu     = float(np.mean(elite_rhos))
            rho_sd     = max(float(np.std(elite_rhos)), 0.05)
            counts     = np.bincount([candidates[i][0] for i in elite_idx],
                                     minlength=self.n_discrete).astype(float)
            probs      = counts / (counts.sum() + 1e-8)

            # Early convergence (Line 44)
            if rho_sd < 0.02:
                break

        return best_d, best_rho, best_res

    @staticmethod
    def _eval_candidate(rho: float, Y: int, res: dict, task: dict) -> float:
        """Algorithm 2, Lines 32-33: compute total reward for a candidate."""
        data     = task['task_size_KB'] * 1000
        cpu      = task['cpu_cycles_per_KB'] * data / 1000
        deadline = task['deadline_ms'] / 1000
        bw_hz    = res['bandwidth_mhz'] * 1e6
        comp_hz  = res['comp_ghz']      * 1e9
        prop     = 0.05 if Y == 0 else 0.0   # cloud propagation delay

        delay = (rho * data) / (bw_hz + 1e-9) \
              + (rho * cpu)  / (comp_hz + 1e-9) \
              + prop

        if delay < deadline:
            revenue = 1.0 - 0.3 * delay / (deadline + 1e-9)
        else:
            revenue = max(0.0, 0.5 - (delay - deadline) / (deadline + 1e-9)) * 0.5

        energy_cost = rho * cpu * 1e-9 * 0.1
        penalty     = 2.0 if delay > deadline else 0.0
        return revenue - energy_cost - penalty

    # ── online training ──────────────────────────────────────────────────────

    def store_transition(self, state: np.ndarray, action: int, rho: float,
                         reward: float, next_state: np.ndarray, done: bool):
        self.replay.append((state.copy(), action, rho, reward,
                            next_state.copy(), float(done)))

    def train_step(self) -> float:
        """Algorithm 2, Lines 51-56: DQN update at end of each user slot."""
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay, self.batch_size)
        loss  = self._loss(batch, self.pdqn)
        self.online_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.pdqn.parameters(), 10.0)
        self.online_opt.step()
        self._step += 1
        if self._step % self.target_update == 0:
            self.target_pdqn.load_state_dict(self.pdqn.state_dict())
        return float(loss.item())


# ============================================================================
# 6.  ENVIRONMENT  (IoT dataset → User-Edge-Cloud simulation)
# ============================================================================

class UserEdgeCloudEnv:
    """
    Simulates K network slices, each with a set of IoT users.
    Slice resources (B_k^T, F_k^T, C_k^T) are controlled by the QMIX agent.
    Task offloading is controlled by the PDQN agent.
    """
    STATE_DIM  = 6   # per-agent observation size (matches ActorNetwork input)
    ACTION_DIM = 3   # per-agent action size       (matches ActorNetwork output)

    def __init__(self, data_path: str, n_slices: int = 3, n_edge: int = 3,
                 users_per_slice: List[int] = None):
        self.n_slices        = n_slices
        self.n_edge          = n_edge
        self.users_per_slice = users_per_slice or [8, 12, 16]

        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} task samples")

        # Total infrastructure resources
        self.total_bw    = 600.0    # MHz
        self.total_edge  = 48.0     # GHz
        self.total_cloud = 80.0     # GHz

        self.slice_bw    = np.zeros(n_slices)
        self.slice_edge  = np.zeros(n_slices)
        self.slice_cloud = np.zeros(n_slices)
        self.tasks       = [[] for _ in range(n_slices)]
        self._idx        = 0
        self.current_step = 0

    def _next_task(self) -> dict:
        row = self.df.iloc[self._idx % len(self.df)]
        self._idx += 1
        server = str(row['server_selection'])
        return {
            'task_size_KB':          float(row['task_size_KB']),
            'deadline_ms':           float(row['deadline_ms']),
            'cpu_cycles_per_KB':     float(row['cpu_cycles_per_KB']),
            'energy_per_cycle_J':    float(row['energy_per_cycle_J']),
            'optimal_offloading_ratio': float(row['offloading_ratio']),
            'optimal_Y':    0 if 'Cloud' in server else 1,
            'system_cost':  float(row['system_cost']),
        }

    def _slice_state(self, k: int) -> np.ndarray:
        """6-dimensional per-slice observation vector."""
        return np.array([
            self.slice_bw[k]    / self.total_bw,
            self.slice_edge[k]  / self.total_edge,
            self.slice_cloud[k] / self.total_cloud,
            self.users_per_slice[k] / 50.0,
            0.5,   # placeholder: avg user utility (updated in step)
            0.1,   # placeholder: failure rate
        ], dtype=np.float32)

    def reset(self) -> List[np.ndarray]:
        self._idx = int(np.random.randint(0, len(self.df)))
        self.current_step = 0

        # Equal initial resource split
        self.slice_bw    = np.ones(self.n_slices) * (self.total_bw    / self.n_slices)
        self.slice_edge  = np.ones(self.n_slices) * (self.total_edge  / self.n_slices)
        self.slice_cloud = np.ones(self.n_slices) * (self.total_cloud / self.n_slices)

        self.tasks = [
            [self._next_task() for _ in range(self.users_per_slice[k])]
            for k in range(self.n_slices)
        ]
        return [self._slice_state(k) for k in range(self.n_slices)]

    def step(self, actions: List[np.ndarray]):
        """
        actions: list of K arrays, each shape (3,) = [bw_frac, edge_frac, cloud_frac].
        Returns: (next_states, profits, done)
        """
        # Apply actions (Algorithm 1, Line 17)
        for k, a in enumerate(actions):
            self.slice_bw[k]    = max(10.0, float(a[0]) * self.total_bw)
            self.slice_edge[k]  = max(1.0,  float(a[1]) * self.total_edge)
            self.slice_cloud[k] = max(5.0,  float(a[2]) * self.total_cloud)

        # Normalise to respect total constraints (C11-C13)
        self.slice_bw    /= self.slice_bw.sum()    / self.total_bw
        self.slice_edge  /= self.slice_edge.sum()  / self.total_edge
        self.slice_cloud /= self.slice_cloud.sum() / self.total_cloud

        # Compute per-slice profits (used in fairness reward)
        profits = []
        for k in range(self.n_slices):
            rev   = 1.0
            r_cost = (self.slice_bw[k]    / self.total_bw    * 10
                    + self.slice_edge[k]  / self.total_edge  * 5
                    + self.slice_cloud[k] / self.total_cloud * 3)
            e_cost = float(np.mean([t['system_cost'] for t in self.tasks[k]])) / 10_000.0
            profit = max(0.0, rev - r_cost - e_cost)
            profits.append(profit)

        # Refresh tasks for next step
        self.tasks = [
            [self._next_task() for _ in range(self.users_per_slice[k])]
            for k in range(self.n_slices)
        ]
        self.current_step += 1
        done = self.current_step >= 50

        next_states = [self._slice_state(k) for k in range(self.n_slices)]
        return next_states, profits, done


# ============================================================================
# 7.  MAIN TRAINING LOOP
# ============================================================================

def main():
    config = {
        'n_slices':        3,
        'n_edge':          3,
        'users_per_slice': [8, 12, 16],
        'total_episodes':  100,
        'steps_per_episode': 20,
        'data_path':       'iot_dataset.csv',
        'fairness_coef':   0.1,
        'fairness_std':    0.05,
        'meta_lr':         1e-3,
    }

    print("=" * 60)
    print("FAIR-QMIX-PDQN-MAML-CEM SYSTEM")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ── Environment ─────────────────────────────────────────────────────────
    env = UserEdgeCloudEnv(
        config['data_path'],
        config['n_slices'],
        config['n_edge'],
        config['users_per_slice']
    )

    # ── Inter-slice agent (Algorithm 1) ─────────────────────────────────────
    slice_agent = FAIR_QMIX_APSN_Agent(
        n_slices    = config['n_slices'],
        state_dim   = UserEdgeCloudEnv.STATE_DIM,    # 6
        action_dim  = UserEdgeCloudEnv.ACTION_DIM,   # 3
        fairness_coef = config['fairness_coef'],
        fairness_std  = config['fairness_std'],
    )

    # ── Intra-slice agent (Algorithm 2) ─────────────────────────────────────
    n_discrete = 2 * config['n_edge']    # Y ∈ {0,1} × x ∈ {1..M}
    off_agent  = MAML_PDQN_CEM_Agent(
        state_dim  = 7,             # (task_size, cpu, deadline, 3×res, step_frac)
        n_discrete = n_discrete,
        n_edge     = config['n_edge'],
        meta_lr    = config['meta_lr'],
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    episode_rewards = []

    for ep in range(config['total_episodes']):
        states = env.reset()   # list of K arrays, each shape (STATE_DIM,)
        ep_rew = 0.0
        slice_loss_val = 0.0
        off_loss_val   = 0.0

        # MAML rapid adaptation at start of each episode
        # (simulates "new slice scheduling slot T" from Algorithm 2, Steps 4-11)
        recent_buf = list(off_agent.replay)[-500:]
        off_agent.rapid_adapt(recent_buf, n_steps=5)

        for step in range(config['steps_per_episode']):

            # ── Inter-slice: select & execute actions (Algorithm 1, Lines 12-19) ──
            actions     = slice_agent.select_action(states, explore=True)
            nxt_states, profits, done = env.step(actions)
            fair_rewards = slice_agent.fairness_reward(profits)

            # Store per-agent transitions
            for k in range(config['n_slices']):
                slice_agent.store_transition(
                    k,
                    states[k],
                    actions[k],
                    fair_rewards[k],
                    nxt_states[k]
                )

            # Train inter-slice agent
            sl = slice_agent.train_step()
            if sl:
                slice_loss_val = sl.get('critic_loss', slice_loss_val)

            # ── Intra-slice: per-user offloading (Algorithm 2, Lines 12-50) ────
            for k in range(config['n_slices']):
                slice_res = {
                    'bandwidth_mhz': env.slice_bw[k],
                    'edge_cpu_ghz':  env.slice_edge[k],
                    'cloud_cpu_ghz': env.slice_cloud[k],
                }
                for task in env.tasks[k]:
                    # 7-dim state: [task_size, cpu, deadline, bw_frac,
                    #               edge_frac, cloud_frac, step_frac]
                    off_state = np.array([
                        task['task_size_KB']      / 1000.0,
                        task['cpu_cycles_per_KB'] / 1000.0,
                        task['deadline_ms']        / 1000.0,
                        env.slice_bw[k]    / env.total_bw,
                        env.slice_edge[k]  / env.total_edge,
                        env.slice_cloud[k] / env.total_cloud,
                        step / config['steps_per_episode'],
                    ], dtype=np.float32)

                    d, rho, _ = off_agent.select_action_with_cem(
                        off_state, slice_res, task)

                    # Reward: closeness to dataset's optimal ratio
                    rew = 1.0 - abs(rho - task['optimal_offloading_ratio'])
                    off_agent.store_transition(
                        off_state, d, rho, rew, off_state, False)

            ol = off_agent.train_step()
            if ol:
                off_loss_val = ol

            ep_rew += sum(fair_rewards)
            states  = nxt_states

            if done:
                break

        episode_rewards.append(ep_rew / config['steps_per_episode'])

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1:3d}/{config['total_episodes']}  "
                f"AvgReward: {episode_rewards[-1]:.4f}  "
                f"SliceLoss: {slice_loss_val:.4f}  "
                f"OffLoss: {off_loss_val:.4f}  "
                f"APSN σ: {[f'{s:.4f}' for s in slice_agent.noise_scales]}"
            )

    print("\n✅ TRAINING COMPLETED")
    print(f"Final avg reward (last 10 eps): {np.mean(episode_rewards[-10:]):.4f}")
    return slice_agent, off_agent, episode_rewards


if __name__ == "__main__":
    slice_agent, off_agent, rewards = main()