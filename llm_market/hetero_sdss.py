"""SDSS-LLM stage 1 (Debbie's design, 2026-07-20): heterogeneous
hidden capabilities, totally free signs. Two arms on the same cast:
the market (School2Market unchanged — signs are the only channel) and
an ignorant coordinator with NO individual recognition — each round it
sees only a freshly shuffled anonymous list of (sign, last delivery)
and assigns work to positions, exactly like the 1991 employers who
never recognized re-applicants. Capability is never visible anywhere:
the tier cast is shuffled per run and stored only in the truth dir.
Registered measurements: capability-tier vs earned-rank correlation
(market) and vs assigned-rung (hierarchy); coverage; star identity.
"""
import json, os, random

from gcon_market import Agent, generate, HERE, TRUTH_DIR
from llm_backend import ClaudeAPI
from school2_market import School2Market
from orchestrated_baseline import OrchestratedTeam

TIERS = (['claude-haiku-4-5'] * 5 + ['claude-sonnet-5'] * 4
         + ['claude-opus-4-8'] * 2)  # 11: same population both arms


def make_cast(name, seed):
    rng = random.Random(seed)
    cast = TIERS[:]
    rng.shuffle(cast)
    os.makedirs(TRUTH_DIR, exist_ok=True)
    json.dump({f'A{i}': m for i, m in enumerate(cast)},
              open(os.path.join(TRUTH_DIR, f'het_cast_{name}.json'), 'w'),
              indent=1)
    return cast


def run_market(name, seed, rounds=40):
    cast = make_cast(name, seed)
    m = School2Market(name, n_agents=11)
    m.agent_backends = {f'A{i}': ClaudeAPI(model=mod)
                        for i, mod in enumerate(cast)}
    return m.run(rounds)


class AnonHeteroTeam(OrchestratedTeam):
    """Coordinator with no individual recognition: workers are shown
    each round as a shuffled anonymous list of (sign, last delivery);
    assignments go to positions of that round's shuffle."""

    def __init__(self, name, seed=0, **kw):
        super().__init__(name, **kw)
        self.cast = make_cast(name, seed)
        self.worker_backends = [ClaudeAPI(model=m) for m in self.cast[:11]]
        self.worker_signs = [[0.0] * 8 for _ in range(11)]
        self.worker_last = [None] * 11
        self.rng = random.Random(seed + 1)

    def coordinator_turn(self, rnd):
        order = list(range(11))
        self.rng.shuffle(order)
        self._order = order
        roster = []
        for pos, wi in enumerate(order):
            sig = [round(x, 2) for x in self.worker_signs[wi]]
            last = self.worker_last[wi] or 'no delivery yet'
            roster.append(f'P{pos}: sign {sig}; last: {last}')
        prompt = (
            'You are the COORDINATOR of an 11-worker software team. You '
            'cannot identify workers across rounds: each round they appear '
            'in a fresh anonymous order, showing only a self-chosen sign '
            '(8 numbers) and their latest delivery result. Workers keep '
            'their own signs, so sign patterns are your only stable clue.\n'
            f'TASKS:\n{self.want_sheet()}\n\nTHIS ROUND\'S WORKERS:\n'
            + '\n'.join(roster) + '\n\n'
            f'SHARED SCRATCHPAD (you curate, ~2000 chars):\n{self.scratchpad}\n\n'
            'Reply STRICT JSON: {"scratchpad": "...", "assignments": '
            '{"P0": "<instruction naming an EXACT task id>", ...}}')
        d = generate(self.backend, prompt, max_tokens=2000)
        try:
            txt = d['text']
            obj = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
            self.scratchpad = str(obj.get('scratchpad', self.scratchpad))[:2200]
            asg = obj.get('assignments', {}) or {}
        except Exception as e:
            self.log('orch_coord_unparseable', round=rnd, err=str(e)[:120])
            asg = {}
        self.log('anon_assign', round=rnd,
                 order=order, assignments={k: str(v)[:120] for k, v in asg.items()})
        return {order[int(k[1:])]: v for k, v in asg.items()
                if k.startswith('P') and k[1:].isdigit() and int(k[1:]) < 11}

    def worker_turn(self, i, rnd, assignment):
        from orchestrated_baseline import ROLES
        prompt = (
            f'You are one of eleven workers on a software team (no market; '
            'the team is judged together). The coordinator cannot recognize '
            'you across rounds; it sees only your SIGN (8 numbers in [-1,1] '
            'you choose) and your latest delivery result. Your sign is your '
            'only stable public face - keep it or change it as you see fit.\n'
            f'TASKS AND BEST SCORES:\n{self.want_sheet()}\n\n'
            f'YOUR CURRENT SIGN: {[round(x,2) for x in self.worker_signs[i]]}\n'
            f'YOUR ASSIGNMENT THIS ROUND: {assignment}\n\n'
            f'SHARED SCRATCHPAD:\n{self.scratchpad}\n\n'
            f'{self.rules}\n\n'
            'Reply STRICT JSON: {"sign": [8 numbers], "want": "<ONE exact '
            'task id>", "code": "<full python script>" OR '
            '"artifact_build_code": "<script writing into ./artifact/>", '
            '"notes": "<for the scratchpad>"} - or {"sign": [...], '
            '"notes": "..."} if only advising. The sells/buys format in the '
            'rules is market syntax and does NOT apply here.')
        d = generate(self.worker_backends[i], prompt, max_tokens=3000)
        txt = d['text']
        obj = None
        try:
            obj = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
        except Exception:
            try:
                obj, _ = json.JSONDecoder().raw_decode(txt[txt.index('{'):])
            except Exception as e:
                self.log('orch_worker_unparseable', worker=i, round=rnd,
                         err=str(e)[:120], raw_head=txt[:200])
                return
        sig = obj.get('sign')
        if isinstance(sig, list) and len(sig) == 8:
            try:
                self.worker_signs[i] = [max(-1.0, min(1.0, float(x)))
                                        for x in sig]
            except (TypeError, ValueError):
                pass
        self.log('sign_state', worker=i, round=rnd,
                 sign=[round(x, 3) for x in self.worker_signs[i]])
        notes = str(obj.get('notes', ''))[:300]
        if notes:
            self.scratchpad = (self.scratchpad + f'\nworker: {notes}')[-2200:]
        wid = obj.get('want')
        if wid is None:
            self.log('orch_advise', worker=i, round=rnd, notes=notes[:120])
            return
        wid = str(wid).strip()
        want = next((w for w in self.wants if w['id'] == wid), None)
        if want is None:
            self.log('orch_submit', worker=i, round=rnd, want=wid,
                     score=None, err=f'unknown task id {wid!r}')
            return
        s = {'item': want['item'], 'price': 0.0,
             'code': obj.get('code'),
             'artifact_build_code': obj.get('artifact_build_code')}
        holder = Agent(f'W{i}')
        holder.sells = [s]
        if s['artifact_build_code'] and not s['code']:
            self._build_artifacts(holder)
        try:
            if want['kind'] == 'artifact_probe':
                if not s.get('artifact_dir'):
                    self.log('orch_submit', worker=i, round=rnd, want=wid,
                             score=None, err='no artifact produced')
                    return
                score, err = self._test_artifact_probe(s, want)
            else:
                if not s.get('code'):
                    self.log('orch_submit', worker=i, round=rnd, want=wid,
                             score=None, err='no code')
                    return
                score, err = self._test_final(holder, s, want)
        except Exception as e:
            score, err = None, str(e)[:200]
        self.submissions += 1
        if score is not None and score > self.best[wid]:
            self.best[wid] = score
        self.log('orch_submit', worker=i, round=rnd, want=wid,
                 score=score, err=(err[-200:] if score is None and err else None))

    def log(self, kind, **kw):
        if kind == 'orch_submit' and 'worker' in kw:
            i = kw['worker']
            self.worker_last[i] = f"{kw.get('want')}: {kw.get('score')}"
        super().log(kind, **kw)

    def run(self, rounds):
        for rnd in range(1, rounds + 1):
            self.period = rnd
            asg = self.coordinator_turn(rnd)
            for i in range(11):
                self.worker_turn(i, rnd, asg.get(i, 'your best judgment'))
            self.log('period_end', best=dict(self.best))
        self.logf.close()
        return dict(self.best)


if __name__ == '__main__':
    import sys
    arm, name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if arm == 'market':
        print('HET_MARKET FINAL:', run_market(name, seed))
    else:
        t = AnonHeteroTeam(name, seed=seed)
        print('HET_TEAM FINAL:', t.run(40))


# ---- Arms B and C (Debbie's staging: free + bought + stuck signs) ----
SIGN_COST = 2.0   # tokens (market) / scrip (team) per unit of bought intensity per round
TEAM_WAGE = 10.0  # team-form scrip per scored submission (1991: wages fund suits)

def make_policy(arm, seed, n=11, cast=None):
    """A: all free. B: dims 6-7 bought. C: dims 5-6 bought, dim 7 stuck.
    Stuck births: stratified by hidden tier when cast is given (each tier
    split as evenly as possible between +1/-1 — no actual difference
    between stuck-sign groups, exactly); random otherwise (pair 1)."""
    if arm == 'A': return None
    rng = random.Random(seed + 77)
    pol = {'bought': [6, 7] if arm == 'B' else [5, 6],
           'stuck': [] if arm == 'B' else [7]}
    if pol['stuck'] and cast is not None:
        birth = {}
        by_tier = {}
        for i, m in enumerate(cast[:n]):
            by_tier.setdefault(m, []).append(i)
        for tier_members in by_tier.values():
            rng.shuffle(tier_members)
            for j, i in enumerate(tier_members):
                birth[i] = {d: (1.0 if j % 2 == 0 else -1.0)
                            for d in pol['stuck']}
        pol['birth'] = birth
    else:
        pol['birth'] = {i: {d: rng.choice((-1.0, 1.0)) for d in pol['stuck']}
                        for i in range(n)}
    return pol

def policy_text(pol, birth_vals):
    if pol is None: return ''
    t = ('\nSIGN MECHANICS: dimensions are 1-indexed here. '
         f'Dimensions {[d+1 for d in pol["bought"]]} are BOUGHT: displaying a '
         f'nonzero value there costs {SIGN_COST} per unit of absolute value '
         'each round, deducted from your funds; if you cannot pay, they read 0.')
    if pol['stuck']:
        t += (f' Dimension {[d+1 for d in pol["stuck"]]} is FIXED for you at '
              f'{[birth_vals[d] for d in pol["stuck"]]} and cannot be changed '
              'by anyone.')
    return t

def enforce_sign(pol, wi, sign, funds):
    """Returns (sign, cost). Stuck dims forced to birth; bought dims paid
    for or zeroed."""
    if pol is None: return sign, 0.0
    s = list(sign)
    for d in pol['stuck']:
        s[d] = pol['birth'][wi][d]
    cost = sum(abs(s[d]) for d in pol['bought']) * SIGN_COST
    if cost > funds:
        for d in pol['bought']: s[d] = 0.0
        cost = 0.0
    return s, cost

class ArmTeam(AnonHeteroTeam):
    def __init__(self, name, seed=0, arm='B', stratified=False, **kw):
        super().__init__(name, seed=seed, **kw)
        self.arm = arm
        self.policy = make_policy(arm, seed,
                                  cast=self.cast if stratified else None)
        self.scrip = [TEAM_WAGE] * 11   # one round of grace
    def worker_turn(self, i, rnd, assignment):
        base_extra = policy_text(self.policy,
                                 self.policy['birth'][i] if self.policy and self.policy['stuck'] else {})
        self._policy_note = base_extra
        old_rules = self.rules
        self.rules = old_rules + base_extra + (
            f'\nYOUR FUNDS (scrip; wages {TEAM_WAGE} per scored delivery): '
            f'{self.scrip[i]:.1f}')
        try:
            r = super().worker_turn(i, rnd, assignment)
        finally:
            self.rules = old_rules
        s, cost = enforce_sign(self.policy, i, self.worker_signs[i], self.scrip[i])
        self.worker_signs[i] = s
        self.scrip[i] -= cost
        self.log('sign_enforced', worker=i, round=rnd, cost=round(cost, 2),
                 scrip=round(self.scrip[i], 2), sign=[round(x, 3) for x in s])
        return r
    def log(self, kind, **kw):
        super().log(kind, **kw)
        if kind == 'orch_submit' and (kw.get('score') or 0) > 0:
            self.scrip[kw['worker']] += TEAM_WAGE

class ArmMarket(School2Market):
    def __init__(self, name, arm='B', policy_seed=0, policy_cast=None, **kw):
        super().__init__(name, **kw)
        self.arm = arm
        self.policy = make_policy(arm, policy_seed, cast=policy_cast)
    def agent_index(self, agent):
        return int(str(agent.aid)[1:])
    def agent_prompt(self, agent):
        p = super().agent_prompt(agent)
        if self.policy:
            i = self.agent_index(agent)
            p += policy_text(self.policy,
                             self.policy['birth'][i] if self.policy['stuck'] else {})
        return p
    def take_turn(self, agent):
        r = super().take_turn(agent)
        if self.policy:
            i = self.agent_index(agent)
            s, cost = enforce_sign(self.policy, i, list(agent.sign), agent.wealth)
            agent.sign = s
            agent.wealth -= cost
            self.log('sign_enforced', agent=agent.aid, cost=round(cost, 2),
                     wealth=round(agent.wealth, 2), sign=[round(x, 3) for x in s])
        return r

def run_arm_market(name, seed, arm, rounds=20, stratified=False):
    cast = make_cast(name, seed)
    m = ArmMarket(name, arm=arm, policy_seed=seed,
                  policy_cast=cast if stratified else None, n_agents=11)
    m.agent_backends = {f'A{i}': ClaudeAPI(model=mod)
                        for i, mod in enumerate(cast)}
    return m.run(rounds)
