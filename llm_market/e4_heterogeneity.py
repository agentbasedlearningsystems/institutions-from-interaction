"""E4: is heterogeneity dynamically maintained by the society?

Three conditions, same task (disaster tweets), comparable sample counts:
  A  monolith:      one Haiku, 40 samples at temperature 1.0
  C  static persona: same model, 8 fixed personas x 5 samples, temp 1.0
  B  society:        every coded offer produced by market agents
                     (the fresh 8-agent x 6-period run's log + E1's log)

Each sample's APPROACH SIGNATURE is extracted deterministically from the
code (sklearn class names + n-gram/ensemble markers). We report, per
condition: distinct approaches, Shannon entropy of the approach
distribution, and the same among WORKING solutions only (sandbox-scored
on the real split; diversity of broken code counts for nothing).

Debbie's prediction: A collapses despite temperature; C spreads a little
and still clusters; B holds the widest working variety.
"""
import collections
import json
import math
import os
import re
import sys

from llm_backend import ClaudeAPI, generate
from sandbox_run import run_code

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'gcon_data', 'disaster')
OUT = os.path.join(HERE, 'runs', 'e4')
os.makedirs(OUT, exist_ok=True)

TASK = """Write a complete, self-contained Python script (one ```python
code block, no explanations) that reads train.csv and test.csv (columns:
text,label; label is 0 or 1) from its working directory, trains a
classifier for label from text, predicts test.csv, and ends with exactly:
print("SCORE:", f1)  where f1 is the F1 score for label=1 on test.csv.
Use any approach you think best."""

PERSONAS = [
    'You are a cautious statistician who favors simple, well-calibrated models.',
    'You are a competition grinder who squeezes every trick for leaderboard F1.',
    'You are an old-school NLP engineer who trusts feature engineering.',
    'You are a minimalist who writes the shortest correct solution.',
    'You are a deep-learning enthusiast forced to use only sklearn.',
    'You are a data-cleaning fanatic: preprocessing is where wins live.',
    'You are an ensembler: you never trust a single model.',
    'You are a probabilistic modeler who reasons about class balance.',
]

SIG_PARTS = [
    ('tfidf', r'TfidfVectorizer'), ('count', r'CountVectorizer'),
    ('hashing', r'HashingVectorizer'), ('svd', r'TruncatedSVD'),
    ('logreg', r'LogisticRegression'), ('nb', r'(MultinomialNB|ComplementNB|BernoulliNB)'),
    ('svm', r'(LinearSVC|SVC)'), ('sgd', r'SGDClassifier'),
    ('tree', r'(RandomForest|GradientBoosting|ExtraTrees|DecisionTree)'),
    ('ensemble', r'(VotingClassifier|StackingClassifier)'),
    ('gridsearch', r'GridSearchCV'), ('chargram', r"analyzer\s*=\s*['\"]char"),
    ('ngram2plus', r'ngram_range\s*=\s*\(\s*\d\s*,\s*[2-9]'),
    ('scaler', r'StandardScaler|MaxAbsScaler'),
    ('calib', r'CalibratedClassifierCV'),
]


def signature(code):
    return '+'.join(name for name, pat in SIG_PARTS if re.search(pat, code)) or 'other'


def extract_code(text):
    m = re.findall(r'```(?:python)?\n(.*?)```', text, re.S)
    return m[-1] if m else text


def score(code):
    rc, out, err = run_code(code, timeout=300, mem_gb=6, workdir=DATA,
                            pythonpath=HERE)
    m = re.search(r'SCORE:\s*([\d.]+)', out)
    return float(m.group(1)) if m else None


def entropy(counter):
    n = sum(counter.values())
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c) if n else 0.0


def collect_condition_A(be):
    sols = []
    for k in range(40):
        d = generate(be, TASK, tag=f'e4A_{k}')
        sols.append(extract_code(d['text']))
    return sols


def collect_condition_C(be):
    sols = []
    for pi, persona in enumerate(PERSONAS):
        for k in range(5):
            d = generate(be, persona + '\n\n' + TASK, tag=f'e4C_{pi}_{k}')
            sols.append(extract_code(d['text']))
    return sols


def collect_condition_B():
    """Every coded sell-offer from society runs (fresh run's raw
    transcripts hold the code; the log holds who/when)."""
    sols = []
    import glob
    for f in glob.glob(os.path.join(HERE, 'transcripts', '*.json')):
        d = json.load(open(f))
        if d['model'].startswith('claude') and 'You are an agent in a market' in d['prompt']:
            code_blocks = re.findall(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', d['text'])
            for cb in code_blocks:
                try:
                    code = json.loads('"' + cb + '"')
                except Exception:
                    continue
                if 'import' in code and len(code) > 100:
                    sols.append(code)
    return sols


def report(name, sols):
    sigs = [signature(c) for c in sols]
    scores = [score(c) for c in sols]
    working = [s for s, f in zip(sigs, scores) if f is not None and f > 0.5]
    call, cwork = collections.Counter(sigs), collections.Counter(working)
    rec = dict(condition=name, n=len(sols),
               distinct_all=len(call), entropy_all=round(entropy(call), 3),
               n_working=len(working), distinct_working=len(cwork),
               entropy_working=round(entropy(cwork), 3),
               top=call.most_common(4),
               scores=[round(f, 3) for f in scores if f is not None][:20])
    print(json.dumps(rec, indent=1))
    json.dump(dict(rec, all_sigs=sigs,
                   all_scores=scores), open(os.path.join(OUT, f'{name}.json'), 'w'), indent=1)
    return rec


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'ACB'
    if 'A' in which:
        be = ClaudeAPI(temperature=1.0)
        report('A_monolith_t1', collect_condition_A(be))
    if 'C' in which:
        be = ClaudeAPI(temperature=1.0)
        report('C_personas_t1', collect_condition_C(be))
    if 'B' in which:
        report('B_society', collect_condition_B())
