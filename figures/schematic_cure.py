"""Vector CURE architecture schematic (Fig. 3a), drawn directly with matplotlib."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.stats import norm

INK = '#333333'
EDGE = '#5f6b73'
FILL = {'chem': '#e3eef8', 'geo': '#e6f0e1', 'sys': '#ece8f5', 'module': '#e5f3f1',
        'learner': '#f4f8fc', 'lik': '#fbe9ec', 'recal': '#eaf2e3', 'prob': '#e8f0f8'}
W, H = 180.0, 60.0          # drawing units follow the panel size in millimetres


def _box(ax, x, y, w, h, fc, lw=.8, r=1.6, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=fc, ec=EDGE, lw=lw, zorder=z))


def _arrow(ax, pts, z=4):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=INK, lw=.8, zorder=z,
            solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=z + 1,
                arrowprops=dict(arrowstyle='-|>,head_length=.45,head_width=.22', color=INK, lw=.8,
                                shrinkA=0, shrinkB=0))


def _tree(ax, cx, cy, color, s=1.0):
    top, left, right = (cx, cy + 2.6 * s), (cx - 2.4 * s, cy - 1.6 * s), (cx + 2.4 * s, cy - 1.6 * s)
    for p in (left, right):
        ax.plot([top[0], p[0]], [top[1], p[1]], color=EDGE, lw=.7, zorder=3)
    for p in (top, left, right):
        ax.add_patch(Circle(p, 1.15 * s, fc=color, ec=EDGE, lw=.6, zorder=4))


def draw_cure_schematic(ax):
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')
    fs = 8.5

    # Predictor groups and shared input bus
    labels = [('Historical\nchemistry', 'chem'), ('Geography', 'geo'), ('Water\nsystem', 'sys')]
    ys = [41.5, 23.5, 5.5]
    for (lab, key), y in zip(labels, ys):
        _box(ax, 1, y, 25, 13.5, FILL[key])
        ax.text(13.5, y + 6.75, lab, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.1)
    bus = 30.5
    for y in ys:
        ax.plot([26, bus], [y + 6.75, y + 6.75], color=INK, lw=.8, zorder=4)
    ax.plot([bus, bus], [ys[-1] + 6.75, ys[0] + 6.75], color=INK, lw=.8, zorder=4)

    # Censored mixture learner module
    _box(ax, 34, 1, 74, 58, FILL['module'], r=2.2, z=0)
    ax.text(71, 55.2, 'Censored mixture learner (CURE)', ha='center', va='center', fontsize=fs, color=INK)
    _box(ax, 37.5, 37, 61, 14.5, FILL['learner'])
    _box(ax, 37.5, 19.5, 61, 14.5, FILL['learner'])
    ax.text(39.5, 46.6, 'Mixture weight', ha='left', va='center', fontsize=fs, color=INK)
    ax.text(39.5, 41.4, 'LightGBM', ha='left', va='center', fontsize=8.5, color='#55606a', gid='keep_colour')
    _tree(ax, 70, 44, '#9cc3e6')
    ax.text(88, 44.2, r'$\pi(x)$', ha='center', va='center', fontsize=9.5, color=INK)
    ax.text(39.5, 29.1, 'Positive\nconcentration', ha='left', va='center', fontsize=fs, color=INK,
            linespacing=1.1)
    ax.text(39.5, 22.6, 'XGBoost-AFT', ha='left', va='center', fontsize=8.5, color='#55606a', gid='keep_colour')
    _tree(ax, 66, 26.6, '#c9a6dc')
    g = np.linspace(-3, 3, 80)
    ax.plot(73.2 + 2.2 * g, 21.5 + 9.2 * norm.pdf(g) / norm.pdf(0), color=INK, lw=.8, zorder=4)
    ax.plot([73.2, 73.2], [21.5, 30.7], color=INK, lw=.6, ls=(0, (2, 1.5)), zorder=4)
    ax.text(89.5, 26.8, r'$\mu(x),\ \sigma$', ha='center', va='center', fontsize=9.5, color=INK)
    _arrow(ax, [(bus, 44.25), (37.5, 44.25)])
    _arrow(ax, [(bus, 26.75), (37.5, 26.75)])

    # Censored likelihood strip
    _box(ax, 37.5, 4, 23.5, 11.5, FILL['lik'])
    ax.text(49.25, 9.75, 'Observed or\nbounded C', ha='center', va='center', fontsize=fs, color=INK, linespacing=1.1)
    _box(ax, 65, 4, 33.5, 11.5, FILL['lik'])
    ax.text(81.75, 11.6, 'Censored likelihood', ha='center', va='center', fontsize=fs, color=INK)
    ax.text(81.75, 7.2, 'EM updates', ha='center', va='center', fontsize=8.5, color='#55606a', gid='keep_colour')
    _arrow(ax, [(61, 9.75), (65, 9.75)])

    # Product junction
    jx, jy = 104, 35.5
    ax.add_patch(Circle((jx, jy), 2.6, fc='white', ec=INK, lw=.8, zorder=5))
    ax.text(jx, jy - .1, r'$\times$', ha='center', va='center', fontsize=10, color=INK, zorder=6)
    _arrow(ax, [(98.5, 44.25), (jx, 44.25), (jx, jy + 2.6)])
    _arrow(ax, [(98.5, 26.75), (jx, 26.75), (jx, jy - 2.6)])

    # Recalibration
    _box(ax, 110, 8, 31, 43.5, FILL['recal'])
    ax.text(125.5, 46.5, 'Recalibration', ha='center', va='center', fontsize=fs, color=INK)
    ax.text(125.5, 41.2, r'shift $\mu$, $\sigma$, $\pi$', ha='center', va='center', fontsize=8.5, color='#55606a', gid='keep_colour')
    g = np.linspace(-3.2, 3.2, 90)
    base = 14
    ax.plot(118.5 + 2.0 * g, base + 16 * norm.pdf(g) / norm.pdf(0), color='#4f7cb3', lw=.9, ls=(0, (3, 1.6)), zorder=4)
    ax.plot(132 + 2.4 * g, base + 13.3 * norm.pdf(g) / norm.pdf(0), color='#c8546a', lw=.9, zorder=4)
    ax.plot([112, 139], [base, base], color=EDGE, lw=.6, zorder=3)
    _arrow(ax, [(122.8, 24), (127.4, 24)])
    _arrow(ax, [(jx + 2.6, jy), (110, jy)])

    # Threshold probabilities
    _box(ax, 143.5, 8, 36, 43.5, FILL['prob'])
    ax.text(161.5, 46.5, 'Threshold probabilities', ha='center', va='center', fontsize=fs, color=INK)
    ax.text(161.5, 41.2, r'$P(C \geq t \mid x)$', ha='center', va='center', fontsize=9, color=INK)
    x0, y0, x1, y1 = 149, 13, 176, 36
    ax.plot([x0, x0], [y0, y1], color=INK, lw=.7, zorder=4)
    ax.plot([x0, x1], [y0, y0], color=INK, lw=.7, zorder=4)
    tt = np.linspace(0, 1, 80)
    for h, k, c in [(.95, 3.0, '#4f7cb3'), (.82, 4.2, '#3f9c8f'), (.66, 6.5, '#9b6bc2')]:
        ax.plot(x0 + .4 + tt * (x1 - x0 - 1), y0 + 1 + h * (y1 - y0 - 2) * np.exp(-k * tt), color=c, lw=.9, zorder=4)
    ax.text((x0 + x1) / 2, y0 - 2.4, 'threshold t', ha='center', va='center', fontsize=8.5, color='#55606a', gid='keep_colour')
    _arrow(ax, [(141, jy), (143.5, jy)])


if __name__ == '__main__':
    # Standalone export of Fig. 3a to the work directory.
    import os, pathlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42})
    out = pathlib.Path(os.environ.get('PFAS_WORK_DIR', pathlib.Path(__file__).resolve().parents[1] / 'work')) / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    draw_cure_schematic(fig.add_axes([0, 0, 1, 1]))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'Fig3a_cure_schematic.{ext}', dpi=600)
    print(out / 'Fig3a_cure_schematic.pdf')
