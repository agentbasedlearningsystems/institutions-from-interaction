"""GCoN Level-1 market: pretrained LLM agents on a blackboard.

The institution (ported from the fleet, now COMPLETE — 2026-07-15 rebuild
after Debbie's audit found the Level-1 port had only half of her design):
  - sign-matched trade that never vetoes; payment on verified delivery;
  - THE quality-market law (no static bar; per-want aspiration; payment
    scaled by score/frontier; audition feedback to every tested seller);
  - roll-order examination (succotash selection: wearing a sign draws
    you into that sign's auditions);
  - RICH DEMAND: a catalog of wants — final products, components, and
    aspirational wants nobody can fill yet ("we want all the demand we
    need, and it doesnt hurt to have it if its not doable");
  - the ARTIFACT ECONOMY (the simulation's pickle economy, ported):
    sellers can produce goods — trained models, embeddings, datasets —
    as FILES, built once, sold as files; buyers' delivered scripts load
    them instead of re-deriving them. Trade in products, not recipes;
  - AGENT-TO-AGENT CLEARING: buy orders actually settle, gated by
    load-verification (her original design gates every purchase), with
    the buyer's notebook as the finer discipline on suppliers;
  - MUTUAL sign matching (sought + displayed, her design; restored by
    her Jul 15 ruling); wealth is PRIVATE (public wealth was a Claude
    invention, ditched by her Jul 15 ruling);
  - private persistent notebooks; token costs; living cost. NO exit:
    bankruptcy was a Claude invention, never Debbie's design (removed
    2026-07-15) — need presses through visible debt, not death; the
    ledger cap is the real spending safety.

Everything is appended to runs/<name>/log.jsonl. All LLM calls go
through the cached, hard-capped backend (llm_backend.generate).
"""
import json
import math
import os
import random
import re
import shutil
import tempfile

from llm_backend import ClaudeAPI, generate
from sandbox_run import run_code
from e4_heterogeneity import signature as code_signature

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- economy
START_WEALTH = 150.0         # runway: races must outlive habituation
TOKEN_COST_IN = 1 / 400.0    # credits per prompt token read
TOKEN_COST_OUT = 1 / 100.0   # credits per token written
TESTS_PER_WANT = 5           # buyer attention budget per want per period (CMA-ES parity: num_tests 5)
BUYERS_WEIGHT = 0.5          # mutual sign blend (her CMA-ES design):
                             # weight on buyer-seeks-seller vs
                             # seller-seeks-buyer
EPS_ROLL = 0.02              # exploration floor in the examination roll
SIGN_SIZE = 8                # signs are float vectors, read and written
                             # as numbers (Debbie's design; the one-to-one
                             # plan translation keeps floats as floats where
                             # no word band applies — the sign is floats).
                             # 8 dims, same as the CMA-ES world (Debbie
                             # Jul 16: same signs both substrates; cosine
                             # reads directions, and 8 dims hold hundreds
                             # of mutually distant regions — the pilots
                             # crowded, they never ran out of room).
                             # Meaning accrues to REGIONS through trade;
                             # nothing is pre-named.

CATALOG = ("shared catalog (ontology) of known parts: pandas, numpy, "
           "sklearn [TfidfVectorizer, CountVectorizer, LogisticRegression, "
           "LinearSVC, MultinomialNB, TruncatedSVD, Pipeline, GridSearchCV, "
           "f1_score], gensim [Word2Vec], joblib. You may petition new "
           "entries via ontology_request.")

# ------------------------------------------------------------------ wants
# The demand catalog. kind='final_f1': deliver a complete script; scored
# by its printed F1 on the want's held-out split. kind='artifact_probe':
# deliver an ARTIFACT (an .npz with float arrays X_train, X_test aligned
# to the want's train/test rows); scored by a fixed probe that trains
# LogisticRegression on (X_train, y_train) and reports held-out accuracy.
# Aspirational wants are posted like any other — unmet demand is the
# standing frontier pull.
def default_wants():
    return [
        dict(id='disaster', kind='final_f1', max_price=40.0,
             item='classifier_text',
             data=os.path.join(HERE, 'gcon_data', 'disaster'),
             text='classifier for disaster tweets: is this tweet about a '
                  'real disaster? deliver a script; F1 on held-out data'),
        dict(id='sentiment', kind='final_f1', max_price=60.0,
             item='classifier_text',
             data=os.path.join(HERE, 'gcon_data', 'sentiment'),
             text='movie-review sentiment classifier (25k reviews; an '
                  'unlabeled.csv with 50k extra reviews is in your data — '
                  'using it well is where the headroom is); deliver a '
                  'script; F1 on held-out data'),
        dict(id='doc_embedding', kind='artifact_probe', max_price=45.0,
             item='vectorSpace_documentEmbedding',
             data=os.path.join(HERE, 'gcon_data', 'sentiment'),
             text='COMPONENT: a document-embedding artifact for the '
                  'sentiment corpus — an .npz file named embedding.npz '
                  'with float arrays X_train, X_test aligned to '
                  'train.csv/test.csv rows; scored by a fixed logistic '
                  'probe trained on your X_train'),
        dict(id='sentiment_ensemble', kind='final_f1', max_price=70.0,
             item='classifier_text_ensemble',
             data=os.path.join(HERE, 'gcon_data', 'sentiment'),
             recipe=dict(require=['ensemble'], min_learners=2),
             text='sentiment classifier that MUST BE an ensemble of at '
                  'least two distinct learner families (the buyer '
                  'specifies the recipe here; deliveries that score well '
                  'but are not ensembles are not bought)'),
        dict(id='sentiment_fewshot', kind='final_f1', max_price=80.0,
             item='classifier_text',
             data=os.path.join(HERE, 'gcon_data', 'sentiment_small'),
             text='ASPIRATIONAL: sentiment classifier trained from only '
                  'the 500 labeled rows in this want\'s train.csv (the '
                  'full unlabeled corpus is fair game), scored on the '
                  'full held-out test. Hard — nobody may do this well '
                  'yet; the price stands anyway'),
    ]


TRUTH_DIR = os.path.join(HERE, 'gcon_truth_k7x2m9')


def truth_labels_path(data_dir):
    """Held-back test labels for a public corpus dir. The public
    test.csv is text-only; graded code never sees labels (Debbie
    2026-07-19 scoring ruling: the market's instruments do the
    scoring, never the graded party)."""
    return os.path.join(TRUTH_DIR, os.path.basename(data_dir),
                        'test_labels.csv')


PROBE_CODE = """
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
d = np.load('artifact/embedding.npz')
ytr = pd.read_csv('train.csv')['label'].values
yte = pd.read_csv({truth_labels!r})['label'].values
clf = LogisticRegression(max_iter=2000).fit(d['X_train'], ytr)
print('SCORE:', accuracy_score(yte, clf.predict(d['X_test'])))
"""

MARKET_LAW = """- The human buyer posts SEVERAL WANTS, each with its own max price, its
  own EXPECTATION (a running average of what it has been getting for
  that want — private, like all its positions), and its own BEST-EVER
  verified score (also private). You learn where you stand from your
  own auditions and sales, nowhere else. Each
  period, for each want, it examines offers in a sign-similarity-weighted
  RANDOM order (closer signs are examined first more often, but everyone
  gets occasional early looks), stops at the first offer beating its
  expectation, and otherwise buys the best working offer it tested.
  Signs route attention; they never veto.
- PAYMENT IS QUALITY-SCALED per want: posted price x (your score / the
  best score ever verified for that want). Matching the best earns full
  price; beating it resets the best and earns full price at the new
  frontier while everyone else's percentage drops. Every tested seller
  is told its score — sale or not.
- ARTIFACTS: you can sell PRODUCED GOODS, not just code. Put
  "artifact_build_code" in a sell entry: a script that writes its
  product files into an out/ directory (it runs ONCE, in a workdir
  containing the datasets under data/<want_id>/). Buyers of your
  artifact receive the FILES, never the build code. Your own delivered
  scripts can load artifacts you have purchased: they appear under
  artifacts/<tag>/ in your script's working directory (your inbox tells
  you each tag).
- AGENT-TO-AGENT TRADE: your BUY orders settle each period against other
  agents' sell offers. Artifact purchases are verified to load before
  payment. Choose suppliers carefully and use your notebook — a bad
  component is your loss.
- A DEMANDED RECIPE BINDS ONLY WHAT YOU DELIVER — one layer deep. How
  you source it is your business: build it yourself or hire (buy
  components and artifacts from other agents) — a specification never
  reaches through your interface into your suppliers. Your sign is what
  carries your whole stack's reputation.
- ALL PRICE TERMS ARE PRIVATE. Your posted prices, your buy budgets,
  and the buyers' budgets never appear on the board — only the clearing
  house sees them, and you learn terms when your own trades settle.
  Offers and wants are visible; positions are not."""

RULES = f"""You are an agent in a market. The rules of the world:
- You interact ONLY through this blackboard. Each turn you see the board
  and return your actions as JSON. Other agents are LLMs like you.
- ORIENTATION — how to read everything you see. Every trade is rendered
  one-to-one in a mixed form: WORDS where a value names a kind of thing,
  NUMBERS where a value is a position or amount. An ITEM (words, e.g.
  classifier_text, vectorSpace_documentEmbedding) is the TYPE of good,
  a path in the shared catalog tree — types must be compatible for a
  trade to be possible at all (a general type accepts any of its
  specializations). A SIGN (8 numbers in [-1,1]) is a POSITION in
  meaning space — positions mean nothing at birth and come to mean what
  is repeatedly traded there; among type-compatible partners, closeness
  of positions is what routes attention. Prices and budgets are numbers
  you never see (terms are private; you learn them when your own trades
  settle). So: the item says WHAT KIND, the sign says WHO/WHERE in the
  culture, and outcomes teach you what everything means.
{MARKET_LAW}
- A delivered script for a 'final' want must be complete Python that
  reads train.csv (columns: text,label) and test.csv (column: text —
  the test labels are HELD BACK by the market) from its working
  directory, trains on train, predicts every test.csv row IN ORDER,
  and writes predictions.csv (single column: label) to its working
  directory. The market scores your predictions against the held-back
  labels itself (F1 for label=1). Printed scores are ignored — the
  instrument does the scoring, never the seller.
- Costs you pay: {TOKEN_COST_IN:.4f} credits per token you read and
  {TOKEN_COST_OUT:.2f} per token you write (thinking is not free).
  Wealth can go negative:
  your debt is private, and it is your need pressing on you — there is
  no exit, and no one will bail you out. Earn.
- {CATALOG}
Return STRICT JSON only, with keys:
  sign: YOUR POSITION in meaning space — a list of 8 numbers in
         [-1, 1]. Signs have NO predefined meaning: regions acquire
         meaning through who trades what there. The buyers' wants sit at
         fixed positions (shown on the board); sellers who deliver a
         want well are found near it, and everyone sees where sales
         happen. Move gradually to keep your reputation's location, or
         jump to reposition. Matching is MUTUAL cosine closeness.
  seeking: a second 8-number vector — what you reach for (customers,
         suppliers, partners). Buy orders may carry their own "sign"
         vector to seek a specific kind of supplier.
  sells: list of {{"price": number,
         "item": str (the catalog type path of what you sell),
         "code": str-or-null, "artifact_build_code": str-or-null}}
         (any compatible buyer, human want or agent, may examine any
         offer: the buyer's test supplies its own data. code = a
         deliverable script. artifact_build_code = a script that
         PRODUCES files into out/ — sold as files.)
  buys: list of {{"want": str (your own words, for your notes),
         (up to 4 sells and 4 buys are read each turn; further
         entries are dropped; a buy without a numeric max_price is
         ignored and you are told)
         "item": str-or-null (the catalog type path you demand, at any
         generality; null = untyped, matches any type),
         "sign": [8 numbers] (where in meaning space you seek it),
         "max_price": number,
         "recipe": null-or-{{"require": [catalog part tokens],
         "min_learners": int}}}} — like the human, you are free to
         specify the recipe of what you buy, at any grain, or not at all
  ontology_request: str or null
  notebook: your PRIVATE notebook, up to 150 words. Replaces last turn's;
         shown back only to you every turn — your only memory beyond
         recent events. Keep your niche, suppliers, prices, plans.
"""


LEARNER_TOKENS = {'logreg', 'nb', 'svm', 'sgd', 'tree'}


def sanitize_code(code):
    """Close the letters-in-parcels channel (Debbie 2026-07-16):
    delivered code components cross the interface stripped of comments
    and docstrings, with locally bound identifiers normalized, so a
    good carries its function, never a message. Imports, attributes,
    and functional string literals are untouched (they are the works).
    Falls back to comment-stripping alone if renaming fails."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    try:
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported.add((a.asname or a.name).split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    imported.add(a.asname or a.name)
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
                for a in (node.args.args + node.args.kwonlyargs):
                    bound.add(a.arg)
        rename = {n: f'c{i}' for i, n in enumerate(sorted(bound - imported))}
        class R(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in rename:
                    node.id = rename[node.id]
                return node
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if node.name in rename:
                    node.name = rename[node.name]
                for a in (node.args.args + node.args.kwonlyargs):
                    if a.arg in rename:
                        a.arg = rename[a.arg]
                return node
        tree = R().visit(tree)
        return ast.unparse(ast.fix_missing_locations(tree))
    except Exception:
        try:
            return ast.unparse(tree)
        except Exception:
            return code


def recipe_ok(code, recipe):
    """The buyer's recipe dial (her SISTER freedom, ported): a want or
    buy order may demand parts of the recipe at any grain; deliveries
    must contain the demanded signature tokens (and, if asked, at least
    min_learners distinct learner families). No recipe = fully general,
    today's default."""
    if not recipe:
        return True, ''
    parts = set(code_signature(code).split('+'))
    missing = [t for t in recipe.get('require', []) if t not in parts]
    if missing:
        return False, f'missing required parts: {missing}'
    n_learners = len(parts & LEARNER_TOKENS)
    if n_learners < recipe.get('min_learners', 0):
        return False, (f'has {n_learners} learner families, recipe '
                       f'demands {recipe["min_learners"]}')
    return True, ''


def sign_cosine(a, b):
    """Cosine similarity between two sign vectors (her design; signs live
    in [-1,1]^8, so cosine genuinely discriminates)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def parse_sign(v, fallback):
    """Accept a list of numbers as a sign; clamp to [-1,1], fix length."""
    if not isinstance(v, list):
        return fallback
    out = []
    for x in v[:SIGN_SIZE]:
        try:
            out.append(max(-1.0, min(1.0, float(x))))
        except (TypeError, ValueError):
            return fallback
    while len(out) < SIGN_SIZE:
        out.append(0.0)
    return out


def render_sign(v, sig=6):
    return '[' + ', '.join(f'{x:.{sig}g}' for x in v) + ']'


def board_precision(vectors, floor=3, cap=17):
    """Adaptive display resolution (Debbie 2026-07-16: 'display more if
    things ever come to the point of that level of significance').
    Start coarse; grow significant digits until every genuinely
    distinct vector in this view renders distinctly, so the board
    reveals exactly as much precision as the society is using and
    never hides emergent fine structure."""
    distinct = {tuple(v) for v in vectors if v}
    for sig in range(floor, cap + 1):
        rendered = {render_sign(list(t), sig) for t in distinct}
        if len(rendered) == len(distinct):
            return sig
    return cap


def seeded_sign(salt):
    rng = random.Random('sign:' + salt)
    return [round(rng.uniform(-1, 1), 3) for _ in range(SIGN_SIZE)]


VALID_ITEMS = {
    # the registry vocabulary: family roots and the catalog's item
    # paths, one-to-one with the world's registry (Debbie 2026-07-16);
    # anything else is not a type, so it is not an item. Replaced by
    # the pinned real-namespace tree when that snapshot is built.
    'classifier', 'classifier_text', 'classifier_text_ensemble',
    'vectorSpace', 'vectorSpace_documentEmbedding',
    'clusterer', 'anomalyDetector', 'preprocessor', 'dataset',
}


def valid_item(item):
    return item if item in VALID_ITEMS else None


def item_compatible(want_item, offer_item):
    """Type compatibility, as in the original design: an offer matches a
    demand when one item path extends the other (a general demand
    accepts any specialization; a specific demand accepts only sellers
    of that specialization or deeper). Missing offer item = the offer
    is untyped and matches only fully-general demands."""
    if not want_item:
        return True
    if not offer_item:
        return False
    w = want_item.lower().split('_stop')[0]
    o = offer_item.lower().split('_stop')[0]
    return o.startswith(w) or w.startswith(o)


def rolled_order(items, weight_of, rng):
    """Succotash selection (Debbie): examination ORDER is sampled by
    similarity weight with an epsilon floor, so even the worst-matched
    seller is occasionally examined first — auditions reach everyone."""
    pool = [(it, weight_of(it) + EPS_ROLL) for it in items]
    out = []
    while pool:
        tot = sum(w for _, w in pool)
        r = rng.uniform(0, tot)
        acc = 0.0
        for i, (it, w) in enumerate(pool):
            acc += w
            if r <= acc:
                out.append(it)
                pool.pop(i)
                break
        else:
            out.append(pool.pop()[0])
    return out


class Agent:
    def __init__(self, aid):
        self.aid = aid
        self.wealth = START_WEALTH
        self.sign = seeded_sign(aid + ':displayed')
        self.seeking = seeded_sign(aid + ':seeking')   # sought sign (mutual)
        self.sells = []
        self.buys = []
        self.history = []      # private inbox: what happened to me
        self.notebook = ''     # private persistent memory (Debbie 2026-07-15)
        self.artifacts = []    # purchased goods: list of (tag, dir)


class Market:
    def __init__(self, name, n_agents=12, model='claude-haiku-4-5',
                 aspiration_alpha=0.05, wants=None, task_type=None,
                 seed_society=None):
        self.name = name
        self.backend = ClaudeAPI(model=model)
        # Mixed-capability societies (SDSS-LLM arms): agent_models maps
        # agent index -> model id, harness-side ONLY. Capability is
        # invisible everywhere: no prompt, board, history, or log body
        # names a model; even the agent does not know its own tier.
        self.agent_backends = {}
        self.agents = [Agent(f'A{i}') for i in range(n_agents)]
        self.period = 0
        self.rundir = os.path.join(HERE, 'runs', name)
        self.artdir = os.path.join(self.rundir, 'artifacts')
        os.makedirs(self.artdir, exist_ok=True)
        self.logf = open(os.path.join(self.rundir, 'log.jsonl'), 'a')
        self.wants = wants if wants is not None else default_wants()
        for w in self.wants:
            w.setdefault('aspiration', 0.0)
            w.setdefault('frontier', 0.0)
            w.setdefault('sign', seeded_sign('want:' + w['id']))
        self.aspiration_alpha = aspiration_alpha
        self._art_n = 0
        # The lazy ontology (LLM world only): agents may already write
        # any software; the catalog is the shared REFERENCE system, and
        # it grows when an agent registers a name — so that two agents
        # both understand what software is being referred to. (In the
        # evolutionary world the ontology is the action space and is
        # given exogenously; it never grows from inside a run.)
        self.registry_extra = []
        self.rules = RULES
        if seed_society:
            import json as _json
            sd = _json.load(open(seed_society))
            pos = sd.get('positions', [])
            for i, a in enumerate(self.agents):
                if pos:
                    p = pos[i % len(pos)]
                    a.sign = parse_sign(p.get('sign'), a.sign)
                    a.seeking = parse_sign(p.get('seeking'), a.seeking)
            practice = sd.get('want_practice', {})
            if practice:
                lines = []
                for wid, share in practice.items():
                    lines.append(f'{wid}: {share}')
                self.rules += (
                    '\nMEASURED PRACTICE of a prior society in this same '
                    'world (its settled trade shares per want; its '
                    'combinations and private knowledge are NOT given): '
                    + ' | '.join(lines) + '\n')
            self.log('seeded_from', seed=seed_society,
                     provenance=sd.get('provenance'))
        if task_type:
            from looking_glass_priors import catalog_for_task
            self.rules = self.rules.replace(CATALOG,
                                            catalog_for_task(task_type))

    def log(self, kind, **kw):
        rec = dict(kind=kind, period=self.period, **kw)
        self.logf.write(json.dumps(rec) + '\n')
        self.logf.flush()

    def _rng(self, salt):
        return random.Random(f'{self.name}:{self.period}:{salt}')

    # ------------------------------------------------------------ prompts
    def board_view(self, viewer=None, k=4):
        # Perception IS the roll (Debbie 2026-07-16: selective pressure
        # as awareness flows through signs stochastically, like
        # everywhere else). The window's slots are sampled by mutual
        # sign closeness with the same floor as every audition; sought
        # vectors, buy orders, descriptions, and the registry feed are
        # not perceivable (CMA-ES parity: agents there see nothing;
        # here, seeing is what happens, so it obeys the same law).
        lines = [f'PERIOD {self.period}']
        shown = [w['sign'] for w in self.wants] + [a.sign for a in self.agents]
        sig = board_precision(shown)
        lines.append('HUMAN BUYER WANTS:')
        for w in self.wants:
            rec = (f' DEMANDED RECIPE: {w["recipe"]}' if w.get('recipe')
                   else '')
            lines.append(f'  [{w["id"]}] item {w.get("item", "")} '
                         f'sign {render_sign(w["sign"], sig)} "{w["text"]}"{rec}')
        others = [a for a in self.agents if a is not viewer]
        if viewer is not None:
            def mutual(a):
                return max(0.0, BUYERS_WEIGHT * sign_cosine(viewer.seeking, a.sign)
                           + (1 - BUYERS_WEIGHT) * sign_cosine(a.seeking, viewer.sign))
            rolled = rolled_order([(mutual(a), a) for a in others],
                                  lambda t: t[0],
                                  self._rng('view:' + viewer.aid))
            others = [a for _, a in rolled[:k]]
            lines.append(f'(you perceive only what your signs reach: '
                         f'{len(others)} positions this turn)')
        for a in others:
            lines.append(f'seller at {render_sign(a.sign, sig)}')
            for s in a.sells:
                has = ('artifact ready' if s.get('artifact_dir') else
                       'builds artifact' if s.get('artifact_build_code') else
                       'with code' if s.get('code') else 'no code yet')
                itm = f' item {s.get("item")}' if s.get('item') else ''
                lines.append(f'   SELLS{itm} ({has}; price private)')
        return '\n'.join(lines)

    def agent_prompt(self, agent):
        inbox = '\n'.join(agent.history[-6:]) or '(nothing yet)'
        notebook = agent.notebook or '(empty — first turn)'
        arts = ', '.join(t for t, _ in agent.artifacts) or '(none)'
        return (self.rules
                + '\n=== THE BOARD ===\n' + self.board_view(viewer=agent)
                + f'\n=== YOUR PRIVATE STATE ({agent.aid}) ===\n'
                + f'wealth: {agent.wealth:.1f}\n'
                + f'artifacts you own (usable by your delivered scripts): {arts}\n'
                + f'recent events:\n' + inbox
                + '\nyour private notebook (as you last wrote it):\n'
                + notebook
                + '\n\nYour JSON actions for this turn:')

    # ------------------------------------------------------------- turns
    def take_turn(self, agent):
        _be = self.agent_backends.get(agent.aid, self.backend)
        d = generate(_be, self.agent_prompt(agent), max_tokens=2000)
        cost = d['tokens_in'] * TOKEN_COST_IN + d['tokens_out'] * TOKEN_COST_OUT
        agent.wealth -= cost
        actions, err = self.parse_actions(d['text'])
        if actions is None:
            d2 = generate(_be,
                          self.agent_prompt(agent)
                          + f'\nYour previous reply was not valid JSON '
                            f'({err}). Reply with STRICT JSON only:',
                          max_tokens=2000)
            agent.wealth -= (d2['tokens_in'] * TOKEN_COST_IN
                             + d2['tokens_out'] * TOKEN_COST_OUT)
            actions, err = self.parse_actions(d2['text'])
        if actions is None:
            agent.history.append(f'period {self.period}: your output was '
                                 f'unparseable; turn lost ({err})')
            self.log('turn_unparseable', agent=agent.aid, err=str(err))
            return
        agent.sign = parse_sign(actions.get('sign'), agent.sign)
        agent.seeking = parse_sign(actions.get('seeking'), agent.seeking)
        agent.sells = [s for s in actions.get('sells', [])
                       if isinstance(s, dict)
                       and isinstance(s.get('price'), (int, float))][:4]
        for s in agent.sells:
            s['item'] = valid_item(s.get('item'))
        for b in agent.buys:
            b['item'] = valid_item(b.get('item'))
        raw_buys = [b for b in actions.get('buys', [])
                    if isinstance(b, dict) and 'want' in b][:4]
        agent.buys = []
        for b in raw_buys:
            if isinstance(b.get('max_price'), (int, float)):
                agent.buys.append(b)
            else:
                agent.history.append(
                    f'period {self.period}: your buy order '
                    f'[{b.get("item") or b.get("want", "?")}] had no '
                    f'numeric max_price and was ignored')
        if actions.get('notebook') is not None:
            agent.notebook = str(actions['notebook'])[:1200]
        req = actions.get('ontology_request')
        if req and isinstance(req, str) and len(req) < 200:
            entry = req.strip()
            if entry and entry not in self.registry_extra:
                self.registry_extra.append(entry)
                agent.history.append(
                    f'period {self.period}: your ontology entry '
                    f'"{entry}" is registered — all agents can now refer '
                    f'to it by this name')
                self.log('ontology_grown', agent=agent.aid, entry=entry)
        self._build_artifacts(agent)
        self.log('turn', agent=agent.aid, wealth=round(agent.wealth, 2),
                 tokens_in=d['tokens_in'], tokens_out=d['tokens_out'],
                 sign=agent.sign,
                 sells=[{k: v for k, v in s.items()
                         if k not in ('code', 'artifact_build_code')}
                        | {'has_code': bool(s.get('code')),
                           'has_artifact': bool(s.get('artifact_dir'))}
                        for s in agent.sells],
                 buys=agent.buys,
                 ontology_request=actions.get('ontology_request'),
                 notebook=agent.notebook)

    @staticmethod
    def parse_actions(text):
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            return None, 'no JSON object found'
        try:
            return json.loads(m.group(0)), None
        except Exception as e:
            return None, str(e)[:120]

    # ---------------------------------------------------------- artifacts
    def _data_workdir(self, base=None):
        """A temp workdir seeded with data/<want_id>/ copies (links)."""
        wd = tempfile.mkdtemp(prefix='gcon_')
        for w in self.wants:
            dst = os.path.join(wd, 'data', w['id'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(w['data']) and not os.path.exists(dst):
                os.symlink(w['data'], dst)
        if base:
            for fn in os.listdir(base):
                p = os.path.join(base, fn)
                if os.path.isfile(p):
                    os.symlink(p, os.path.join(wd, fn))
        return wd

    def _build_artifacts(self, agent):
        """Run any new artifact_build_code ONCE; archive out/ as the good.
        The build is verified non-empty before the offer goes live."""
        for s in agent.sells:
            if not s.get('artifact_build_code') or s.get('artifact_dir'):
                continue
            wd = self._data_workdir()
            os.makedirs(os.path.join(wd, 'out'), exist_ok=True)
            rc, out, err = run_code(s['artifact_build_code'], timeout=300,
                                    mem_gb=6, workdir=wd, pythonpath=HERE)
            outdir = os.path.join(wd, 'out')
            files = [f for f in os.listdir(outdir)] if os.path.isdir(outdir) else []
            if files:
                self._art_n += 1
                tag = f'good_p{self.period}_n{self._art_n}'
                dst = os.path.join(self.artdir, tag)
                shutil.copytree(outdir, dst)
                s['artifact_dir'] = dst
                s['artifact_tag'] = tag
                agent.history.append(
                    f'period {self.period}: your artifact [{s.get("item", "untyped")}] '
                    f'built OK ({", ".join(files[:4])}) — on sale as {tag}')
                self.log('artifact_built', agent=agent.aid, tag=tag,
                         files=files[:8], item=s.get('item'))
            else:
                agent.history.append(
                    f'period {self.period}: your artifact build FAILED '
                    f'(no files in out/). stderr: {(err or "")[-200:]}')
                self.log('artifact_build_failed', agent=agent.aid,
                         item=s.get('item'), err=(err or '')[-300:])
            shutil.rmtree(wd, ignore_errors=True)

    def _verify_artifact_loads(self, artdir):
        """Institutional gate on agent purchases: the good must at least
        load (files exist, readable). Finer quality is the buyer's own
        judgment and notebook — bounded rationality with a floor."""
        try:
            files = os.listdir(artdir)
            return bool(files)
        except OSError:
            return False

    # ------------------------------------------------- agent-to-agent trade
    def agent_clearing(self):
        """Buy orders settle: type gate + sign roll, verified delivery,
        midpoint price. Each buy row may fill once per period (her
        SISTER buy rows; the one-purchase-per-buyer throttle was the
        analyst's, removed on her 2026-07-16 ruling)."""
        for buyer in rolled_order([a for a in self.agents if a.buys],
                                  lambda a: 1.0, self._rng('buyers')):
            for b in buyer.buys:
                cands = []
                for seller in self.agents:
                    if seller is buyer:
                        continue
                    for s in seller.sells:
                        if not (s.get('artifact_dir') or s.get('code')):
                            continue
                        if s['price'] > b.get('max_price', 0):
                            continue
                        if not item_compatible(b.get('item'), s.get('item')):
                            continue
                        sought = parse_sign(b.get('sign'), buyer.seeking)
                        toward = sign_cosine(sought, seller.sign)
                        back = sign_cosine(seller.seeking, buyer.sign)
                        sim = (BUYERS_WEIGHT * toward
                               + (1 - BUYERS_WEIGHT) * back)
                        if sim > -1:
                            cands.append((max(sim, 0.0), seller, s))
                for sim, seller, s in rolled_order(
                        cands, lambda c: c[0], self._rng('clear:' + buyer.aid)):
                    if b.get('recipe'):
                        basis = (s.get('artifact_build_code')
                                 if s.get('artifact_dir') else s.get('code'))
                        ok, why = recipe_ok(basis or '', b['recipe'])
                        if not ok:
                            continue
                    if s.get('artifact_dir'):
                        if not self._verify_artifact_loads(s['artifact_dir']):
                            continue
                        tag = s['artifact_tag']
                        buyer.artifacts.append((tag, s['artifact_dir']))
                        delivery = (f'artifact {tag} — your delivered '
                                    f'scripts find it under artifacts/{tag}/')
                    else:
                        self._art_n += 1
                        tag = f'good_p{self.period}_n{self._art_n}_c'
                        dst = os.path.join(self.artdir, tag)
                        os.makedirs(dst, exist_ok=True)
                        with open(os.path.join(dst, 'component.py'), 'w') as f:
                            f.write(sanitize_code(s['code']))
                        buyer.artifacts.append((tag, dst))
                        delivery = (f'code component {tag} — file '
                                    f'artifacts/{tag}/component.py')
                    mid = round((b.get('max_price', s['price'])
                                 + s['price']) / 2.0, 2)
                    buyer.wealth -= mid
                    seller.wealth += mid
                    buyer.history.append(
                        f'period {self.period}: you BOUGHT [{s.get("item", "untyped")}] '
                        f'from the seller at {render_sign(seller.sign)} '
                        f'for {mid} — {delivery}')
                    seller.history.append(
                        f'period {self.period}: the buyer at '
                        f'{render_sign(buyer.sign)} bought your '
                        f'[{s.get("item", "untyped")}] for {mid}')
                    self.log('agent_trade', buyer=buyer.aid,
                             seller=seller.aid, paid=mid,
                             item=s.get('item'), tag=tag)
                    break

    # ------------------------------------------------------ human buying
    def _test_final(self, seller, s, want):
        wd = self._data_workdir(base=want['data'])
        for tag, adir in seller.artifacts:
            dst = os.path.join(wd, 'artifacts', tag)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.lexists(dst):   # repeat purchases of the same
                os.symlink(adir, dst)      # artifact are legitimate

        rc, out, err = run_code(s['code'], timeout=300, mem_gb=6,
                                workdir=wd, pythonpath=HERE)
        # Held-out scoring (Debbie 2026-07-19): the instrument computes
        # the score from the seller's predictions and the held-back
        # labels; nothing the graded code prints is trusted.
        try:
            import pandas as pd
            from sklearn.metrics import f1_score
            preds = pd.read_csv(os.path.join(wd, 'predictions.csv'))['label']
            yte = pd.read_csv(truth_labels_path(want['data']))['label']
            if len(preds) != len(yte):
                shutil.rmtree(wd, ignore_errors=True)
                return None, (f'predictions.csv has {len(preds)} rows; '
                              f'test has {len(yte)}')
            score = float(f1_score(yte, preds, pos_label=1))
        except Exception as e:
            shutil.rmtree(wd, ignore_errors=True)
            return None, (err[-150:] + ' | ' if err else '') + \
                'no scoreable predictions.csv: ' + str(e)[:150]
        shutil.rmtree(wd, ignore_errors=True)
        return score, None

    def _test_artifact_probe(self, s, want):
        wd = self._data_workdir(base=want['data'])
        os.symlink(s['artifact_dir'], os.path.join(wd, 'artifact'))
        probe = PROBE_CODE.format(truth_labels=truth_labels_path(want['data']))
        rc, out, err = run_code(probe, timeout=300, mem_gb=6,
                                workdir=wd, pythonpath=HERE)
        shutil.rmtree(wd, ignore_errors=True)
        m = re.search(r'SCORE:\s*([\d.]+)', out)
        return (float(m.group(1)) if m else None), err

    def _settle(self, want, a, s, score):
        want['frontier'] = max(want['frontier'], score)
        # Midpoint settlement (her double auction): the meeting point of
        # two PRIVATE positions, then quality-scaled.
        base = (want['max_price'] + s['price']) / 2.0
        paid = round(base * (score / want['frontier']), 2)
        note = (f'paid {paid} = midpoint x (score {score:.3f} / '
                f'best-ever {want["frontier"]:.3f}) for [{want["id"]}]')
        want['aspiration'] += self.aspiration_alpha * (score - want['aspiration'])
        a.wealth += paid
        a.history.append(
            f'period {self.period}: human BOUGHT your [{s.get("item", "untyped")}] — {note}')
        self.log('sale', want=want['id'], seller=a.aid, price=s['price'],
                 paid=paid, score=score,
                 aspiration=round(want['aspiration'], 4),
                 frontier=round(want['frontier'], 4))

    def human_buy(self):
        for want in self.wants:
            offers = []
            for a in self.agents:
                for s in a.sells:
                    if not item_compatible(want.get('item'), s.get('item')):
                        continue
                    deliverable = (s.get('artifact_dir')
                                   if want['kind'] == 'artifact_probe'
                                   else s.get('code'))
                    if deliverable and s['price'] <= want['max_price']:
                        toward = sign_cosine(want['sign'], a.sign)
                        back = sign_cosine(a.seeking, want['sign'])
                        sim = BUYERS_WEIGHT * toward + (1 - BUYERS_WEIGHT) * back
                        offers.append((sim, a, s))
            best = None
            tested = 0
            for sim, a, s in rolled_order(offers, lambda o: o[0],
                                          self._rng('want:' + want['id'])):
                if tested >= TESTS_PER_WANT:
                    self.log('attention_budget_hit', want=want['id'],
                             tested=tested, offered=len(offers))
                    break
                tested += 1
                if want['kind'] == 'artifact_probe':
                    score, err = self._test_artifact_probe(s, want)
                else:
                    score, err = self._test_final(a, s, want)
                self.log('delivery_test', want=want['id'], seller=a.aid,
                         sign_sim=round(sim, 3), price=s['price'],
                         score=score,
                         sig=(code_signature(s['code']) if s.get('code') else None),
                         err=(err[-300:] if score is None else None))
                if score is None:
                    a.history.append(
                        f'period {self.period}: your [{want["id"]}] delivery '
                        f'FAILED to run; no payment. '
                        f'stderr: {(err or "")[-200:]}')
                    continue
                if want['kind'] == 'final_f1' and want.get('recipe'):
                    ok, why = recipe_ok(s['code'], want['recipe'])
                    if not ok:
                        a.history.append(
                            f'period {self.period}: your [{want["id"]}] '
                            f'delivery scored {score:.3f} but did NOT '
                            f'match the demanded recipe ({why}); not '
                            f'bought.')
                        self.log('recipe_rejected', want=want['id'],
                                 seller=a.aid, score=score, why=why)
                        continue
                if best is None or score > best[0]:
                    best = (score, a, s)
                if score >= want['aspiration']:
                    break
                a.history.append(
                    f'period {self.period}: tested for [{want["id"]}], '
                    f'score={score:.3f} — below expectation '
                    f'{want["aspiration"]:.3f}; the buyer kept looking. '
                    f'(Beat best-ever {want["frontier"]:.3f} for full price.)')
            if best is not None:
                score, a, s = best
                self._settle(want, a, s, score)
            else:
                self.log('no_sale', want=want['id'])

    # -------------------------------------------------------------- run
    def run(self, periods):
        for _ in range(periods):
            self.period += 1
            # Activation order: seeded uniform shuffle per period,
            # matching the CMA-ES world's StagedActivation(shuffle=True).
            order = [a for _, a in rolled_order(
                [(1.0, a) for a in self.agents], lambda t: t[0],
                self._rng('activation'))]
            for a in order:
                self.take_turn(a)
            self.agent_clearing()
            self.human_buy()
            self.log('period_end',
                     wealth={a.aid: round(a.wealth, 2) for a in self.agents})
        self.logf.close()
        return {a.aid: round(a.wealth, 2) for a in self.agents}


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    periods = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    name = sys.argv[3] if len(sys.argv) > 3 else f'market_{n}a{periods}p'
    m = Market(name, n_agents=n, task_type='text-classification')
    final = m.run(periods)
    print('final wealth:', final)
    print('log:', os.path.join('runs', name, 'log.jsonl'))
