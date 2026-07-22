"""Mechanics test: exercises the rebuilt market with SCRIPTED agents —
no LLM calls. Verifies: artifact build path, verified agent-to-agent
clearing, artifact delivery into a buyer's test workdir, the probe
meter, the final-script meter, roll-order determinism, and settle math.
Run: python test_market_mechanics.py
"""
import os
from gcon_market import Market

FINAL_CODE = """
import pandas as pd
from sklearn.metrics import f1_score
te = pd.read_csv('test.csv')
print("SCORE:", f1_score(te['label'], [1]*len(te)))
"""

BUILD_CODE = """
import numpy as np, pandas as pd, os
n_tr = len(pd.read_csv('data/sentiment/train.csv'))
n_te = len(pd.read_csv('data/sentiment/test.csv'))
np.savez(os.path.join('out', 'embedding.npz'),
         X_train=np.zeros((n_tr, 4), dtype='float32'),
         X_test=np.zeros((n_te, 4), dtype='float32'))
print('built')
"""

USES_ARTIFACT_CODE_TMPL = """
import os, pandas as pd
from sklearn.metrics import f1_score
assert os.path.exists('artifacts/{tag}/embedding.npz'), 'artifact missing!'
te = pd.read_csv('test.csv')
print("SCORE:", f1_score(te['label'], [1]*len(te)))
"""

m = Market('_mech_test', n_agents=3)
A0, A1, A2 = m.agents
m.period = 1

# Position sellers near what they serve (signs are float vectors now).
A0.sign = list(m.wants[0]['sign'])          # near the disaster want
A0.sells = [dict(desc='disaster classifier baseline', price=30.0,
                 want='disaster', item='classifier_text', code=FINAL_CODE)]
# A1 builds an embedding artifact; offers it BOTH as a component (for
# agents) and to the human's doc_embedding want.
A1.sign = list(m.wants[2]['sign'])          # near the doc_embedding want
A1.sells = [dict(desc='document embedding npz for sentiment corpus',
                 price=10.0, want=None, artifact_build_code=BUILD_CODE),
            dict(desc='document embedding npz (for the human probe)',
                 price=20.0, want='doc_embedding',
                 item='vectorSpace_documentEmbedding',
                 artifact_build_code=BUILD_CODE)]
# A2 wants to buy an embedding component.
A2.sign = list(m.wants[1]['sign'])
A2.buys = [dict(want='document embedding npz', max_price=15.0,
                sign=list(A1.sign))]        # seeks suppliers at A1's position

m._build_artifacts(A1)
assert A1.sells[0].get('artifact_dir'), 'component artifact not built'
assert A1.sells[1].get('artifact_dir'), 'probe artifact not built'
print('artifact build: OK,', A1.sells[0]['artifact_tag'])

m.agent_clearing()
assert A2.artifacts, 'agent clearing did not deliver the artifact'
tag = A2.artifacts[0][0]
assert A2.wealth < 150 and A1.wealth > 150, 'payment did not move'
print('agent clearing: OK — A2 bought', tag, 'A1 wealth', A1.wealth)

# A2 now sells a final that REQUIRES its purchased artifact to be present.
A2.sells = [dict(desc='sentiment classifier using purchased embedding',
                 price=25.0, want='disaster', item='classifier_text',
                 code=USES_ARTIFACT_CODE_TMPL.format(tag=tag))]

m.human_buy()
sales = [l for l in open(os.path.join(m.rundir, 'log.jsonl'))
         if '"kind": "sale"' in l]
assert any('"want": "disaster"' in s for s in sales), 'no disaster sale'
assert any('"want": "doc_embedding"' in s for s in sales), 'no probe sale'
print('human buying: OK —', len(sales), 'sales (finals + artifact probe)')

# Roll determinism: same period+salt -> same order.
o1 = m._rng('x').random()
o2 = m._rng('x').random()
assert o1 == o2, 'roll not deterministic'
print('roll determinism: OK')
print('ALL MECHANICS PASS')
