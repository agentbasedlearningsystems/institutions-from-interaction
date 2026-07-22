"""Orchestrated baseline (2026-07-19): the standard multi-agent recipe —
a coordinator, assigned roles, a shared scratchpad — pointed at the SAME
wants, corpora, and measuring instruments as the school societies, at the
same total turn budget (12 agents x N rounds). No market: no signs, no
attention, no prices, no buying or selling between agents. Deliveries are
submitted directly and scored by the same tests. The comparison this
enables: emergent market coordination versus hand-orchestrated
coordination, same production technology, same budget, same judge.
"""
import json
import os

from gcon_market import Agent, generate, run_code, HERE
from school2_market import School2Market

ROLES = [
    'data-cleaning specialist (normalization, dedup, corpus prep)',
    'embedding specialist (representations: tfidf, SVD, umap, gensim)',
    'embedding specialist (a second, working independently)',
    'classifier builder (disaster corpus focus)',
    'classifier builder (reviews corpus focus)',
    'classifier builder (AI-text-detection focus)',
    'ensemble engineer (combining distinct learner families)',
    'few-shot specialist (small-label-budget training)',
    'cross-corpus generalist (one model, two domains)',
    'reviewer (reads teammates code, suggests fixes)',
    'toolsmith (shared utilities for the scratchpad)',
]


class OrchestratedTeam(School2Market):
    """Same wants and testers as School2Market; orchestration instead of
    a market. run(rounds): each round = 1 coordinator turn + 11 worker
    turns; submissions tested immediately; best scores shared."""

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.scratchpad = ('(empty — coordinator: use this space for '
                          'shared plans, working code snippets, results)')
        self.best = {w['id']: 0.0 for w in self.wants}
        self.submissions = 0

    def want_sheet(self):
        rows = []
        for w in self.wants:
            rows.append(f"- {w['id']} (pays up to {w['max_price']}, "
                        f"best so far {self.best[w['id']]:.3f}): "
                        f"{w['text'][:180]}")
        return '\n'.join(rows)

    def coordinator_turn(self, rnd):
        prompt = (
            'You are the COORDINATOR of a software team of 11 specialists. '
            'The team is judged on the tasks below (same scoring as an '
            'open market elsewhere; higher scores on more tasks = better).\n'
            f'TASKS:\n{self.want_sheet()}\n\n'
            f'TEAM ROLES:\n' + '\n'.join(
                f'W{i}: {r}' for i, r in enumerate(ROLES)) + '\n\n'
            f'SHARED SCRATCHPAD (you curate it, max ~2000 chars):\n'
            f'{self.scratchpad}\n\n'
            'Reply STRICT JSON: {"scratchpad": "<updated shared notes>", '
            '"assignments": {"W0": "<specific instruction>", ...}} '
            'Assign every worker something concrete this round. When an '
            'assignment targets a task, name its EXACT task id (submissions '
            'under any other name are dropped).')
        d = generate(self.backend, prompt, max_tokens=2000)
        try:
            txt = d['text']
            obj = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
            self.scratchpad = str(obj.get('scratchpad', self.scratchpad))[:2200]
            asg = obj.get('assignments', {}) or {}
        except Exception as e:
            self.log('orch_coord_unparseable', round=rnd, err=str(e)[:120])
            asg = {}
        self.log('orch_assign', round=rnd,
                 assignments={k: str(v)[:160] for k, v in asg.items()})
        return asg

    def worker_turn(self, i, rnd, assignment):
        prompt = (
            f'You are W{i}, the {ROLES[i]} on an 11-specialist software '
            'team (no market — you deliver work, the team is judged '
            'together).\n'
            f'TASKS AND CURRENT BEST SCORES:\n{self.want_sheet()}\n\n'
            f'YOUR ASSIGNMENT THIS ROUND: {assignment}\n\n'
            f'SHARED SCRATCHPAD:\n{self.scratchpad}\n\n'
            f'{self.rules}\n\n'
            'To submit work reply STRICT JSON: {"want": "<ONE exact task '
            'id from the list>", '
            '"code": "<full python script>" OR "artifact_build_code": '
            '"<script that writes artifact files into ./artifact/>", '
            '"notes": "<for the scratchpad>"} — or {"notes": "..."} '
            'alone if only advising. IMPORTANT: the sells/buys action '
            'format in the rules above is market-world syntax and does '
            'NOT apply here — there is no trading on this team; use ONLY '
            'the {"want", "code", "notes"} format.')
        d = generate(self.backend, prompt, max_tokens=3000)
        txt = d['text']
        obj = None
        try:
            obj = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
        except Exception:
            try:
                obj, _ = json.JSONDecoder().raw_decode(txt[txt.index('{'):])
            except Exception as e:
                self.log('orch_worker_unparseable', worker=i, round=rnd,
                         err=str(e)[:120], raw_head=txt[:250])
                return
        notes = str(obj.get('notes', ''))[:300]
        if notes:
            self.scratchpad = (self.scratchpad + f'\nW{i}: {notes}')[-2200:]
        wid = obj.get('want')
        if wid is None and isinstance(obj.get('sells'), list):
            # Safety net for the format collision: a worker replying in
            # the market's sells syntax is still submitting work.
            for sell in obj['sells']:
                if isinstance(sell, dict) and (sell.get('code') or
                                               sell.get('artifact_build_code')):
                    w = next((w for w in self.wants
                              if w['item'] == sell.get('item')), None)
                    if w is not None:
                        wid = w['id']
                        obj.setdefault('code', sell.get('code'))
                        obj.setdefault('artifact_build_code',
                                       sell.get('artifact_build_code'))
                        break
        if wid is None:
            self.log('orch_advise', worker=i, round=rnd, notes=notes[:120],
                     raw_head=(txt[:200] if not notes else None))
            return
        wid = str(wid).strip()
        want = next((w for w in self.wants if w['id'] == wid), None)
        if want is None:
            self.log('orch_submit', worker=i, round=rnd, want=wid,
                     score=None, err=f'unknown task id {wid!r}')
            self.scratchpad = (self.scratchpad +
                               f'\nSYSTEM: W{i} submission DROPPED - '
                               f'{wid!r} is not a task id; use exactly one '
                               'id from the task list.')[-2200:]
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

    def run(self, rounds):
        for rnd in range(1, rounds + 1):
            self.period = rnd
            asg = self.coordinator_turn(rnd)
            for i in range(len(ROLES)):
                self.worker_turn(i, rnd, asg.get(f'W{i}', 'continue your '
                                                 'best current work'))
            self.log('period_end', best=dict(self.best))
        self.logf.close()
        return dict(self.best)
