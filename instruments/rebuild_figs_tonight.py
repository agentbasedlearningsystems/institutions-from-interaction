"""Rebuild fig_division.pdf and fig_employment.pdf from tonight's
(July 21 late-evening) pooled instrument reads. Values hand-carried
from merged_division.out and the employment sections of refresh/*.out."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = './'
BLUE = '#3d76af'

# world -> (trades, evenness) merged census, >=12 distinct qualify
division = [
    ('basic-227', 20, 0.99), ('basic-224', 19, 0.97), ('basic-229', 18, 0.96),
    ('basic-220', 17, 0.95), ('basic-231', 17, 0.96), ('alwayson-212', 17, 0.97),
    ('basic-228', 16, 0.95), ('basic-230', 16, 0.91), ('basic-226', 16, 0.99),
    ('basic-222', 16, 0.98), ('alwayson-210', 15, 0.94), ('basic-225', 15, 0.98),
    ('basic-223', 15, 0.98), ('basic-219', 13, 0.97), ('alwayson-211', 13, 0.98),
]
fig, ax = plt.subplots(figsize=(7.2, 2.4))
xs = range(len(division))
ax.bar(xs, [d[1] for d in division], color=BLUE)
for i, (w, t, e) in enumerate(division):
    ax.text(i, t + 0.3, f'{e:.2f}', ha='center', va='bottom',
            fontsize=6, rotation=90)
ax.set_xticks(list(xs))
ax.set_xticklabels([d[0] for d in division], rotation=55, ha='right',
                   fontsize=6.5)
ax.set_ylabel('distinct trades', fontsize=8)
ax.set_ylim(0, 23)
ax.tick_params(labelsize=7)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + 'fig_division.pdf')

# world -> end-window learner-to-learner value share (%), all 27 measured
emp = [
    ('alwayson-212', 19.76), ('basic-235', 10.64), ('basic-220', 8.52),
    ('basic-219', 6.57), ('basic-231', 4.96), ('basic-232', 4.72),
    ('basic-230', 3.84), ('basic-224', 3.65), ('basic-226', 3.09),
    ('basic-228', 2.14), ('basic-223', 1.88), ('alwayson-210', 1.52),
    ('basic-229', 1.40), ('basic-222', 1.40), ('basic-227', 1.27),
    ('alwayson-211', 0.0), ('alwayson-213', 0.0), ('alwayson-214', 0.0),
    ('alwayson-215', 0.0), ('basic-216', 0.0), ('basic-217', 0.0),
    ('basic-218', 0.0), ('basic-221', 0.0), ('basic-225', 0.0),
    ('basic-233', 0.0), ('basic-234', 0.0), ('basic-236', 0.0),
]
fig2, ax2 = plt.subplots(figsize=(7.2, 2.2))
xs2 = range(len(emp))
ax2.bar(xs2, [e[1] for e in emp], color=BLUE)
ax2.set_xticks(list(xs2))
ax2.set_xticklabels([e[0] for e in emp], rotation=55, ha='right',
                    fontsize=6)
ax2.set_ylabel('learner-to-learner share\nof end-window payment value',
               fontsize=7.5)
ax2.set_ylim(0, 21)
ax2.tick_params(labelsize=7)
for s in ('top', 'right'):
    ax2.spines[s].set_visible(False)
fig2.tight_layout(pad=0.4)
fig2.savefig(OUT + 'fig_employment.pdf')
print('both figures rebuilt')
