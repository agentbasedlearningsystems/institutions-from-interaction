"""School v2: the bazaar plus Debbie's complex-demand rungs (2026-07-19).

Two additions to the twelve-rung ladder, mirroring the evolutionary
staged-waves design:
  - a DEDUPLICATION end task ("almost just data cleaning"): the corpus
    holds 25 disguised duplicate rows (inflections a lemmatizer undoes,
    injected stopwords a stopword-remover undoes, punctuation pads a
    stripper undoes, light char noise as residual difficulty — the
    disguiser calibrated on the evolutionary side); sellers deliver the
    list of (planted_row, original_row) pairs; scored by F1 against the
    hidden key, which lives OUTSIDE the seller-visible data directory.
  - a CROSS-CORPUS generalization rung: one delivered classifier script
    scored on TWO corpora; the score is the WORSE of the two F1s, so
    only genuine generalization pays.
Prep is repriced to 40 — the specialist rung it turned out to be.
"""
import os

from bazaar_market import BazaarMarket, bazaar_wants
from gcon_market import HERE

from gcon_market import TRUTH_DIR
DEDUP_KEY_PATH = os.path.join(TRUTH_DIR, 'sentiment_dedup_key.json')

DEDUP_PROBE_CODE = f"""
import csv, json
key = json.load(open({DEDUP_KEY_PATH!r}))
truth = {{(int(k['planted_row']), int(k['original_row'])) for k in key}}
claimed = set()
try:
    with open('artifact/pairs.csv') as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            if len(row) >= 2:
                try:
                    claimed.add((int(float(row[0])), int(float(row[1]))))
                except ValueError:
                    pass
except FileNotFoundError:
    pass
tp = len(claimed & truth)
prec = tp / len(claimed) if claimed else 0.0
rec = tp / len(truth) if truth else 0.0
f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
print('SCORE:', round(f1, 6))
"""


def school2_wants():
    D = lambda name: os.path.join(HERE, 'gcon_data', name)
    wants = bazaar_wants()
    for w in wants:
        if w['id'].startswith('prep'):
            # The lesson of school v1: downstream-lift cleaning is expert
            # work; price it as the specialist rung it is.
            w['max_price'] = 40.0
    wants.append(dict(
        id='dedup_reviews', kind='artifact_probe', probe='dedup',
        max_price=30.0, item='preprocessor_dedupList',
        data=D('sentiment_dedup'),
        text='END TASK, cleaning-centered: this corpus contains 25 rows '
             'that are DISGUISED DUPLICATES of earlier rows (reworded '
             'padding, inserted filler words, inflected word forms, '
             'doubled characters). Deliver artifact/pairs.csv with '
             'columns planted_row,original_row (0-based indices into '
             'train.csv) listing each disguised copy and its original. '
             'Scored by F1 against the true planted list. Normalization '
             'is the whole game: strip padding, remove filler words, '
             'undo word inflections, collapse doubled letters, then '
             'match.'))
    wants.append(dict(
        id='clf_cross', kind='final_f1', max_price=75.0,
        item='classifier_text', data=D('sentiment'),
        data2=D('disaster'),
        text='GENERALIZATION RUNG: one delivered classifier script is '
             'trained and scored on TWO corpora separately (movie '
             'reviews AND disaster tweets); your score is the WORSE of '
             'the two F1s. Only programs that genuinely generalize '
             'across domains pay.'))
    return wants


class School2Market(BazaarMarket):
    def __init__(self, name, **kw):
        kw.setdefault('wants', school2_wants())
        super().__init__(name, **kw)
        self.rules += (
            '\nNEW RUNGS: dedup_reviews pays for finding disguised '
            'duplicate rows (deliver artifact/pairs.csv: '
            'planted_row,original_row); clf_cross pays the WORSE of two '
            'corpora F1s — generalize or earn less.\n')

    def _test_artifact_probe(self, s, want):
        if want.get('probe') != 'dedup':
            return super()._test_artifact_probe(s, want)
        import shutil
        from gcon_market import run_code
        wd = self._data_workdir(base=want['data'])
        os.symlink(s['artifact_dir'], os.path.join(wd, 'artifact'))
        rc, out, err = run_code(DEDUP_PROBE_CODE, timeout=120, mem_gb=2,
                                workdir=wd, pythonpath=HERE)
        shutil.rmtree(wd, ignore_errors=True)
        import re as _re
        m = _re.search(r'SCORE:\s*([\d.]+)', out)
        return (float(m.group(1)) if m else None), err

    def _test_final(self, seller, s, want):
        if not want.get('data2'):
            return super()._test_final(seller, s, want)
        s1, e1 = super()._test_final(seller, s, dict(want, data=want['data'], data2=None))
        if s1 is None:
            return None, e1
        s2, e2 = super()._test_final(seller, s, dict(want, data=want['data2'], data2=None))
        if s2 is None:
            return None, e2
        return min(s1, s2), None
