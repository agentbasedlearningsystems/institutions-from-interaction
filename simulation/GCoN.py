"""Generative Cooperative Network (GCoN) agent — transformer-based SISTER.

Drops into SnetSim as a direct replacement for SISTER. Each agent holds its
own small transformer that proposes a float vector; the existing ontology
decoder (float_vec_to_trade_plan) converts it to a trade plan. Trained with
REINFORCE against the same per-step fitness signal that CMA-ES uses.

Design choices, all modifiable via agent_parameters in the scenario config:
  d_model:  64   transformer hidden dim (default small)
  n_layers: 2    transformer encoder layers
  n_heads:  4    attention heads
  lr:       3e-4 Adam learning rate
  baseline_momentum: 0.9  exponential moving average for advantage baseline
  init_log_sigma: -1.0    initial log std of proposal distribution (≈0.37)
  grad_clip: 1.0         gradient clipping

The context the transformer sees each step is a sequence of token embeddings,
one per agent on the blackboard, where each token is the agent's displayed
sign concatenated with a flag indicating whether it is 'self'. After the
transformer pass, the pooled representation drives two heads: the proposal
mean (1148-dim) and a per-dim log-sigma. The proposal is then sampled from a
Gaussian, clipped to [0, 1], and decoded.
"""
import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation.SnetAgent import SnetAgent


DEVICE = torch.device("cuda" if torch.cuda.is_available() and
                     not os.environ.get("GCON_CPU") else "cpu")


class _AgentPolicy(nn.Module):
    """One transformer policy network per agent.

    SISTER-compliant design: the agent has NO fixed identity. The base
    vector is a *learnable* parameter, randomly initialized at agent
    creation (matching CMA-ES initial distribution-mean sampling) and
    evolved by REINFORCE — analogous to CMA-ES's per-agent distribution
    mean evolving across generations. The transformer reads the
    blackboard (peers' signs + own is_self flag) and emits a context-
    dependent perturbation; the final proposal is
        mean = sigmoid(base_logit + perturbation_scale * delta).
    Both base_logit and the transformer parameters update together.

    Earlier versions of this file froze base_logit as a buffer, giving
    agents permanent identity not consistent with SISTER (where agents
    are distinguished only by what they post to the blackboard). The
    learnable variant preserves the bootstrapping advantage of a
    non-uniform initial proposal without freezing identity.

    Input sequence: (N_agents, sign_size + 1) — one token per agent on
    blackboard (sign + is_self flag).
    Output: (mean_vec in [0, 1], per-dim log_sigma).
    """
    def __init__(self, sign_size, vector_size, d_model=64, n_layers=2,
                 n_heads=4, init_log_sigma=-1.0, perturbation_scale=0.5,
                 freeze_base_vector=False, mean_pool_readout=False):
        super().__init__()
        self.sign_size = sign_size
        self.vector_size = vector_size
        self.d_model = d_model
        self.perturbation_scale = perturbation_scale
        self.mean_pool_readout = mean_pool_readout

        # Base vector: random init in [0.1, 0.9] matching CMA-ES initial
        # mean, stored in logit space. Default learnable (SISTER-compliant:
        # evolves through REINFORCE analogously to CMA-ES per-agent
        # distribution mean). With freeze_base_vector=True it is registered
        # as a buffer instead — used for reproducing the paper's earlier
        # experiments where the base vector was treated as fixed identity.
        base_unit = torch.rand(vector_size) * 0.8 + 0.1
        if freeze_base_vector:
            self.register_buffer("base_logit", torch.logit(base_unit))
        else:
            self.base_logit = nn.Parameter(torch.logit(base_unit))

        self.token_embed = nn.Linear(sign_size + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.delta_head = nn.Linear(d_model, vector_size)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.log_sigma = nn.Parameter(torch.full((vector_size,), init_log_sigma))

    def forward(self, tokens, self_idx=None):
        """tokens: (1, N_agents, sign_size + 1). Returns (mean_vec, log_sigma).

        Default readout is the self-token (the agent's own attention-
        contextualized representation). Mean-pooling washes out the
        agent-specific perspective with increasing N: the self-token
        contributes ~1/N to the pooled representation, leaving each agent
        with an increasingly homogenized signal. Self-token extraction
        lets each agent's transformer encode "what should I propose given
        how peers look from MY position." Set ``mean_pool_readout=True``
        for the original mean-pool behavior (used by paper-reported runs).
        """
        h = self.token_embed(tokens)
        h = self.encoder(h)
        if self.mean_pool_readout or self_idx is None or self_idx >= h.shape[1]:
            pooled = h.mean(dim=1)
        else:
            pooled = h[:, self_idx, :]
        delta = self.delta_head(pooled).squeeze(0)             # logit-space delta
        mean = torch.sigmoid(self.base_logit + self.perturbation_scale * delta)
        return mean, self.log_sigma


class GCoN(SnetAgent):
    """Transformer-based SISTER. Same interface; different brain."""

    def __init__(self, unique_id, model, message, parameters):
        super().__init__(unique_id, model, message, parameters)

        p = self.parameters or {}
        self.d_model = int(p.get("d_model", 64))
        self.n_layers = int(p.get("n_layers", 2))
        self.n_heads = int(p.get("n_heads", 4))
        self.lr = float(p.get("lr", 1e-4))
        self.baseline_momentum = float(p.get("baseline_momentum", 0.95))
        self.init_log_sigma = float(p.get("init_log_sigma", -0.4))
        self.grad_clip = float(p.get("grad_clip", 0.5))
        self.entropy_coef = float(p.get("entropy_coef", 0.01))
        self.min_log_sigma = float(p.get("min_log_sigma", -1.5))
        self.perturbation_scale = float(p.get("perturbation_scale", 0.5))
        self.freeze_base_vector = bool(p.get("freeze_base_vector", False))
        self.mean_pool_readout = bool(p.get("mean_pool_readout", False))
        self.warmstart_from = p.get("warmstart_from")  # path to CMA-ES dump dir

        self.vector_size_ = self.vector_size()
        self.sign_size = self.p["sign_size"]

        # Policy network lives on device
        self.policy = _AgentPolicy(
            sign_size=self.sign_size,
            vector_size=self.vector_size_,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            init_log_sigma=self.init_log_sigma,
            perturbation_scale=self.perturbation_scale,
            freeze_base_vector=self.freeze_base_vector,
            mean_pool_readout=self.mean_pool_readout,
        ).to(DEVICE)

        # Warmstart from a CMA-ES dump if specified. Each agent loads its
        # own per-agent file; the loaded vector becomes the initial
        # base_logit. A transformer warm-started this way begins at the
        # CMA-ES plateau and refines through REINFORCE rather than from
        # uniform-random init, matching the intent of CR data absorption.
        if self.warmstart_from:
            try:
                import json as _json
                wpath = os.path.join(self.warmstart_from, f"agent_{unique_id}.json")
                if os.path.exists(wpath):
                    with open(wpath) as f:
                        wdata = _json.load(f)
                    mean_vec = torch.tensor(wdata["mean"], dtype=torch.float32,
                                            device=DEVICE)
                    # Match length; pad/clip if sizes differ
                    if mean_vec.numel() == self.vector_size_:
                        mean_vec = mean_vec.clamp(0.001, 0.999)
                        with torch.no_grad():
                            new_logit = torch.logit(mean_vec)
                            if isinstance(self.policy.base_logit, torch.nn.Parameter):
                                self.policy.base_logit.data.copy_(new_logit)
                            else:
                                # buffer (frozen mode) — just overwrite the buffer
                                self.policy.base_logit.copy_(new_logit)
                        print(f"GCoN agent {unique_id}: warmstarted from {wpath}")
            except Exception as e:
                print(f"GCoN agent {unique_id}: warmstart failed: {e}")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # Baseline for variance reduction (EMA of returns)
        self.baseline = 0.0
        self.step_count = 0

        # Initialize: propose, post to blackboard
        self.initial_trade_plan = None
        if not message:
            initial_vec, log_prob = self._sample_proposal()
            self.initial_vec = initial_vec.detach().cpu().numpy().astype(float)
            self.message, float_vec = self.float_vec_to_trade_plan(self.initial_vec)
        else:
            self.initial_trade_plan = copy.deepcopy(self.message)
            mask = copy.deepcopy(self.initial_trade_plan)
            initial_vec, log_prob = self._sample_proposal()
            self.initial_vec = initial_vec.detach().cpu().numpy().astype(float)
            self.message, float_vec = self.float_vec_to_trade_plan(self.initial_vec, mask)

        self.float_vec = float_vec
        self.last_log_prob = log_prob  # saved for REINFORCE update next step
        self.model.blackboard.append(self.message)

        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0

        print(f"IN GCoN init,{self.b[self.unique_id]['label']} (device={DEVICE})")

    # -- Context construction ------------------------------------------------

    def _build_context(self):
        """(1, N_agents, sign_size + 1) tensor: each token = [sign, is_self].

        During __init__ of the first agent, the blackboard is empty; in that
        case we emit a single zero-sign self token so the transformer has
        something to attend to. Once all agents have posted, the blackboard
        has one token per agent.
        """
        tokens = []
        for i, msg in enumerate(self.b):
            sign = msg.get("sign") or [0.0] * self.sign_size
            sign = list(sign)[: self.sign_size]
            sign = sign + [0.0] * (self.sign_size - len(sign))
            is_self = 1.0 if i == self.unique_id else 0.0
            tokens.append(sign + [is_self])
        if not tokens:
            tokens = [[0.0] * self.sign_size + [1.0]]  # self-only fallback
        return torch.tensor(tokens, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    def _sample_proposal(self):
        """Returns (vec, log_prob). vec is a tensor of shape (vector_size,).

        log_prob is the MEAN log-prob across dimensions (not sum). For a
        1148-D Normal, .sum() produces O(1000)-scale losses that destabilize
        REINFORCE; .mean() keeps loss magnitude O(1) at the cost of slower
        convergence, which is the right tradeoff for a high-dim continuous
        policy with sparse reward.

        Passes self_idx so the policy can extract the encoded representation
        at the agent's own token position rather than mean-pooling.
        """
        context = self._build_context()
        # The agent's position in the encoded sequence equals its index in
        # self.b (the blackboard list), unless the blackboard is empty in
        # which case _build_context emits a self-only fallback.
        if context.shape[1] >= len(self.b):
            self_idx = self.unique_id
        else:
            self_idx = 0
        mean, log_sigma = self.policy(context, self_idx=self_idx)
        sigma = log_sigma.exp()
        dist = torch.distributions.Normal(mean, sigma)
        raw = dist.rsample()
        vec = raw.clamp(0.0, 1.0)
        log_prob = dist.log_prob(raw).mean()
        # Save the raw (unclamped) action for the PPO surrogate to
        # recompute its log-prob next step. detach() so the saved
        # tensor doesn't keep the current graph alive.
        self.last_action_raw = raw.detach()
        return vec, log_prob

    # -- Mesa / SnetSim interface -------------------------------------------

    def step(self):
        print(f"IN GCoN step,{self.b[self.unique_id]['label']} t={self.model.schedule.time}")

        # 1. Compute reward for the proposal posted last step.
        result = (
            self.agiTokens * self.parameters["fitness_weights"]["agi_tokens"]
            + self.max_buyer_score * self.parameters["fitness_weights"]["buyer_score"]
            + self.max_seller_score * self.parameters["fitness_weights"]["seller_score"]
        )
        bought_items = self.get_bought_items()
        self.model.print_reproduction_report_line(self, result, bought_items)

        # 2. Policy update. Default: REINFORCE with entropy bonus.
        # If ppo_clip_eps > 0, use a single-trajectory PPO surrogate
        # (track an "old" log-prob from the action sample, recompute
        # the current log-prob, clip the importance ratio). This is a
        # MARL baseline comparison requested by reviewers; in the
        # 1-sample-per-step setting the clip mostly stabilizes
        # large-magnitude updates without changing the gradient sign.
        advantage = result - self.baseline
        # log_sigma floor keeps the policy exploratory
        with torch.no_grad():
            self.policy.log_sigma.clamp_(min=self.min_log_sigma)
        entropy = self.policy.log_sigma.sum()  # Normal entropy ~ log(sigma) + const

        ppo_eps = float(self.parameters.get("ppo_clip_eps", 0.0) or 0.0)
        if ppo_eps > 0 and getattr(self, "last_action_raw", None) is not None:
            # PPO surrogate: importance-weighted advantage with clip.
            # Recompute current log-prob of the SAME action that was
            # sampled last step (stored in self.last_action_raw).
            ctx = self._build_context()
            self_idx = self.unique_id if ctx.shape[1] >= len(self.b) else 0
            mean, log_sigma = self.policy(ctx, self_idx=self_idx)
            sigma = log_sigma.exp()
            dist = torch.distributions.Normal(mean, sigma)
            new_logp = dist.log_prob(self.last_action_raw).mean()
            ratio = torch.exp(new_logp - self.last_log_prob.detach())
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - ppo_eps, 1 + ppo_eps) * advantage
            loss = -torch.min(unclipped, clipped) - self.entropy_coef * entropy
        else:
            loss = -(self.last_log_prob * advantage) - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        # 3. Baseline EMA
        self.baseline = self.baseline_momentum * self.baseline \
                      + (1 - self.baseline_momentum) * result

        # 4. Log metrics periodically
        self.step_count += 1
        popsize = int(self.parameters.get("num_chromosomes", 25))
        if hasattr(self.model, "metrics") and self.step_count % popsize == 0:
            # Mirror SISTER's per-generation log. We don't have a population,
            # so pass a singleton result list. sigma reported as mean of
            # exp(log_sigma) across dims.
            mean_sigma = float(self.policy.log_sigma.exp().mean().item())
            self.model.metrics.log_generation(self, [result], mean_sigma)

        # 5. Mask (if this agent has a scheduled initial message pattern)
        mask = None
        if self.initial_trade_plan:
            step = math.floor(self.model.schedule.time)
            im = self.initial_trade_plan.get("initial_message", 0)
            fm = self.initial_trade_plan.get("final_message", sys.maxsize)
            mp = self.initial_trade_plan.get("message_period", 1)
            if im <= step <= fm and step % mp == 0:
                mask = copy.deepcopy(self.initial_trade_plan)

        # 6. Propose next float vec, decode, post to blackboard.
        next_vec, next_log_prob = self._sample_proposal()
        # Use float64 + convert to Python floats to avoid JSON-serialize issues
        vec_np = next_vec.detach().cpu().numpy().astype(float)
        new_message, float_vec = self.float_vec_to_trade_plan(vec_np, mask)

        # 7. Reset per-step reward accumulators and replace message.
        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0
        self.set_message(new_message)
        self.float_vec = float_vec
        self.last_log_prob = next_log_prob

    def buyer_score_notification(self, score, tradenum):
        if self.max_buyer_score < score:
            self.max_buyer_score = score

    def seller_score_notification(self, score, tradenum):
        if self.max_seller_score < score:
            self.max_seller_score = score

    def payment_notification(self, agiTokens, tradenum):
        self.agiTokens += agiTokens
