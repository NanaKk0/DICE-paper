"""Approved six-panel properties/efficiency layout from archived observations."""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt, font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

WIDTH_MM, HEIGHT_MM = 180.0, 180.0
BLUE, RED, GREEN, GOLD, PURPLE = '#009eaf', '#e53935', '#13aa7a', '#e7a817', '#9162d6'


def style():
    font_manager.findfont('Arial', fallback_to_default=False)
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'legend.frameon': False, 'axes.linewidth': .7, 'axes.spines.top': False,
        'axes.spines.right': False, 'lines.linewidth': 1.0, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
        'mathtext.cal': 'Arial'})


def axis(fig, x, top, width, height):
    ax = fig.add_axes([x / WIDTH_MM, 1 - (top + height) / HEIGHT_MM,
        width / WIDTH_MM, height / HEIGHT_MM])
    ax.tick_params(which='major', length=2.8, width=.7, pad=2)
    ax.tick_params(which='minor', length=1.5, width=.55, pad=2)
    return ax


def legend(fig, handles, centre_mm, top_mm, columns, **kwargs):
    return fig.legend(handles=handles, loc='upper center', ncol=columns,
        bbox_to_anchor=(centre_mm / WIDTH_MM, 1 - top_mm / HEIGHT_MM),
        borderaxespad=0, handlelength=1.3, handletextpad=.45, columnspacing=.8,
        labelspacing=.4, **kwargs)


def eos(fig, plotter):
    data = plotter.csv('data/physical_validation/AL_EOS_PLOT.csv')
    ax = axis(fig, 17, 21, 65, 33)
    phases = [('alpha', r'$\alpha$', BLUE), ('delta_jp50s1', r'$\delta$-JP50s1', GREEN),
        ('gamma_digne', r'$\gamma$-Digne', GOLD), ('kappa', r'$\kappa$', RED),
        ('theta', r'$\theta$', PURPLE)]
    assert len(data) == 45 and set(data.phase) == {p[0] for p in phases}
    handles = []
    for name, label, colour in phases:
        q = data[data.phase.eq(name)].sort_values('V_fu')
        assert len(q) == 9
        ax.plot(q.V_fu / 5, q.MLIP_relative_meV_atom, color=colour)
        ax.plot(q.V_fu / 5, q.DFT_relative_meV_atom, 'o', ms=3, mfc='white',
            mec=colour, mew=.9, linestyle='none')
        handles.append(Line2D([], [], color=colour, label=label))
    ax.set(xlabel=r'Volume (Å$^3$ atom$^{-1}$)',
        ylabel=r'$E-E_{\alpha,\mathrm{ref}}$ (meV atom$^{-1}$)', ylim=(0, 100))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks([8.0, 8.5, 9.0, 9.5, 10.0])
    legend(fig, [Line2D([], [], color='#353535', label='DICE'),
        Line2D([], [], color='#353535', marker='o', mfc='white', ls='none', ms=3.7, label='DFT')],
        49.5, 3, 2)
    # Keep each phase row centred independently, as in the approved figure.
    legend(fig, handles[:3], 49.5, 10.1, 3)
    legend(fig, handles[3:], 49.5, 14.9, 2)
    return ax, len(data)


def liquid(fig, plotter):
    data = plotter.csv('data/physical_validation/AL_G_figure_source_data.csv')
    axes, counts = [], {}
    for kind, top, name in [('X', 21, 'X-ray'), ('N', 41, 'Neutron')]:
        ax = axis(fig, 110, top, 65, 13)
        q = data[data.kind.eq(kind) & data.r_A.ge(1.5) & data.r_A.le(10)].sort_values('r_A')
        assert len(q) == 851
        ax.fill_between(q.r_A, q.G_mean - q.G_sd, q.G_mean + q.G_sd,
            color=BLUE, alpha=.16, linewidth=0)
        ax.plot(q.r_A, q.G_mean, color=BLUE)
        ax.plot(q.r_A, q.G_experiment, color='#353535', ls='--', lw=.9)
        ax.text(.64, .68, name, transform=ax.transAxes, fontsize=8)
        ax.set(xlim=(1.5, 10), ylim=(-.15, 3.05), ylabel=r'$G(r)$')
        ax.set_xticks([2, 4, 6, 8, 10]);ax.set_yticks([0, 1, 2, 3])
        if kind == 'X':
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r'$r$ (Å)', labelpad=2)
        axes.append(ax);counts[kind] = len(q)
    legend(fig, [Line2D([], [], color=BLUE, label='DICE'),
        Line2D([], [], color='#353535', ls='--', label='Experiment')], 142.5, 3, 2)
    return axes, counts


def conductivity(fig, plotter):
    data = plotter.csv('data/physical_validation/YSZ_CONDUCTIVITY.csv')
    ax = axis(fig, 17, 82, 65, 30)
    ax.set_gid('c-conductivity-panel')
    for comp, colour, marker in [('8YSZ', BLUE, 'o'), ('10YSZ', RED, 's')]:
        q = data[data.simulation_label.eq(comp)].sort_values('T_K')
        assert len(q) == 4
        assert (q[['simulation_NE_mean_S_cm', 'experiment_S_cm']] > 0).all().all()
        x = q.T_K
        ax.errorbar(x, q.simulation_NE_mean_S_cm, yerr=q.simulation_NE_SD_S_cm,
            fmt=marker, color=colour, ms=4, capsize=2, elinewidth=.7)
        ax.scatter(x, q.experiment_S_cm, marker=marker, s=23, facecolors='none',
            edgecolors=colour, linewidths=1)
    ax.set(xlabel=r'$T$ (K)', ylabel=r'$\sigma$ (S cm$^{-1}$)',
        xlim=(1000 / .65, 1000 / .85), yscale='log', ylim=(.035, .5))
    ax.set_xticks([1500, 1400, 1300, 1200])
    ax.yaxis.set_major_locator(ticker.FixedLocator([.05, .1, .2, .4]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(['0.05', '0.1', '0.2', '0.4']))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # Independently centre both rows and align them with the two rows above d.
    source_key = legend(fig, [
        Line2D([], [], color='black', marker='o', mfc='black', mec='black',
            ls='none', ms=3.7, label='DICE'),
        Line2D([], [], color='black', marker='o', mfc='none', mec='black',
            ls='none', ms=3.7, label='Experiment'),
    ], 49.5, 69, 2)
    source_key.set_gid('c-source-legend')
    composition_key = legend(fig, [
        Line2D([], [], color=BLUE, marker='o', ls='none', ms=3.7, label='8YSZ'),
        Line2D([], [], color=RED, marker='s', ls='none', ms=3.7, label='10YSZ'),
    ], 49.5, 74.2, 2)
    composition_key.set_gid('c-composition-legend')
    return ax, {'shown_8YSZ_10YSZ': 8, '12YSZ_retained_in_SI': 4}


def adsorption(fig, plotter):
    data = plotter.csv('data/physical_validation/MIL_MEAN_SD.csv')
    pars = plotter.csv('data/physical_validation/CHEN_PARAMETERS.csv').set_index('temperature_K')
    ax = axis(fig, 110, 82, 65, 30)
    handles = []
    for temp, colour, marker in [(298, BLUE, 'o'), (308, RED, 's'), (318, GREEN, '^')]:
        q = data[data.temperature_K.eq(temp)].sort_values('pressure_bar')
        assert len(q) == 11 and (q.n_repeats == 3).all()
        ax.errorbar(q.pressure_bar, q.uptake_mean_mmol_g, yerr=q.uptake_sd_mmol_g,
            color=colour, marker=marker, ms=3, capsize=1.8, elinewidth=.7)
        x = np.linspace(0, 1, 301)
        qs, b = pars.loc[temp, ['qs_mmol_g', 'b_per_bar']]
        ax.plot(x, qs * b * x / (1 + b * x), color=colour, ls='--', lw=.95)
        np.testing.assert_allclose(q.experiment_fit_mmol_g,
            qs * b * q.pressure_bar / (1 + b * q.pressure_bar), atol=1e-10)
        handles.append(Line2D([], [], color=colour, marker=marker, label=f'{temp} K', ms=3))
    ax.set(xlabel=r'CO$_2$ pressure (bar)', ylabel=r'CO$_2$ uptake (mmol g$^{-1}$)',
        xlim=(0, 1.02), ylim=(0, 4))
    ax.set_yticks([0, 1, 2, 3, 4])
    legend(fig, [Line2D([], [], color='#353535', marker='o', label='DICE', ms=3),
        Line2D([], [], color='#353535', ls='--', label='Experiment')], 142.5, 69, 2)
    legend(fig, handles, 142.5, 74.2, 3)
    return ax, len(data)


def efficiency(fig, plotter):
    data = plotter.csv('data/performance/THREE_SYSTEM_LATENCY.csv')
    blocks = plotter.csv('data/performance/THREE_SYSTEM_TIMING_BLOCKS.csv')
    guides = plotter.csv('data/performance/OOM_DISPLAY_GUIDES.csv')
    ax = axis(fig, 17, 140, 65, 28.5)
    ax.set_xscale('log');ax.set_yscale('log')
    systems = [('al2o3', r'Al$_2$O$_3$', 'o', '-'), ('ysz', 'YSZ', '^', '--'),
        ('mil', 'MIL-120(Al)', 's', '-.')]
    assert len(data) == 83 and len(blocks) == 415 and len(guides) == 6
    assert set(data.system) == {x[0] for x in systems}, set(data.system)
    for system, label, marker, ls in systems:
        for model, colour in [('Neo', RED), ('Nano', BLUE)]:
            q = data[data.system.eq(system) & data.model.eq(model)].sort_values('atoms')
            assert len(q) and (q[['atoms', 'latency_ms_q25', 'latency_ms_median', 'latency_ms_q75']] > 0).all().all()
            y = q.latency_ms_median.to_numpy()
            ax.errorbar(q.atoms, y,
                yerr=np.vstack([y-q.latency_ms_q25.to_numpy(),q.latency_ms_q75.to_numpy()-y]),
                color=colour, marker=marker, ls=ls, ms=2.6, mfc='white' if model=='Neo' else colour,
                mew=.7, lw=.95, elinewidth=.55, capsize=1)
            guide, = guides[guides.system.eq(system) & guides.model.eq(model)].to_dict('records')
            assert guide['last_measured_atoms'] == q.atoms.iloc[-1]
            ax.plot([q.atoms.iloc[-1],guide['OOM_atoms_observed']],
                [y[-1],guide['display_only_endpoint_ms']], color=colour, ls=':', lw=.9)
            ax.plot(guide['OOM_atoms_observed'],guide['display_only_endpoint_ms'],
                'x', color='#222222', ms=4.1, mew=1.0)
    ax.set(xlim=(7.5, 3e7), ylim=(.25, 2000),
        xlabel='Number of atoms', ylabel='Time per MD step (ms)')
    ax.set_xticks([1e1, 1e3, 1e5, 1e7])
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10,numticks=5))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter());ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    # Preserve the approved same-960-atom dimension bracket in panel e only.
    q = data[data.system.eq('al2o3') & data.atoms.eq(960)].set_index('model')
    lower, upper = q.loc['Nano','latency_ms_median'], q.loc['Neo','latency_ms_median']
    assert 195 < upper/lower < 205
    ax.plot([1040, 1350, 1350, 1040], [lower, lower, upper, upper], color='#777777', lw=.6, zorder=0)
    ax.text(1700, np.sqrt(lower*upper), '≈200×', va='center', ha='left', fontsize=8, weight='bold')
    handles = [Line2D([],[],color='#353535',marker=m,ls=ls,label=label,ms=3)
        for _,label,m,ls in systems]
    handles.append(Line2D([],[],color='#222222',marker='x',ls='none',label='OOM',ms=4))
    system_key = legend(fig, handles, 49.5, 128, 4)
    system_key.set_gid('e-systems-legend')
    return ax, {'latency_points':len(data),'timing_blocks':len(blocks),'OOM_guides':len(guides),
        'same_960_atom_ratio':float(upper/lower)}


def exploration(fig, plotter):
    data = plotter.csv('data/sampling_costs/DEVELOPMENT_SAMPLING_COST.csv').set_index('system').loc[['Al2O3/AL3','YSZ/AL2','MOF-MIL-120']]
    assert len(data) == 3 and (data[['student_recorded_h','teacher_estimated_h']] > 0).all().all()
    np.testing.assert_allclose(data.student_percent_of_teacher,
        100*data.student_recorded_h/data.teacher_estimated_h)
    # Slightly taller than e, with the same baseline and a closer colour key.
    ax = axis(fig, 110, 136.5, 65, 32)
    x = np.arange(3)
    for offset, column, colour in [(-.18,'student_recorded_h',BLUE),(.18,'teacher_estimated_h',RED)]:
        bars = ax.bar(x+offset, data[column], width=.31, color=colour)
        for bar,value in zip(bars,data[column]):
            ax.annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,value),
                xytext=(0,2),textcoords='offset points',ha='center',va='bottom',fontsize=8)
    # Author decision: retain measured/estimated hour values, remove all f ratios.
    ax.set(xticks=x,xticklabels=[r'Al$_2$O$_3$','YSZ','MIL-120(Al)'],
        ylabel='Exploration time (h)',ylim=(0,350))
    ax.set_yticks([0,100,200,300])
    return ax, data.reset_index().to_dict('records')


def draw(plotter):
    style()
    fig = plotter.start(HEIGHT_MM)
    _, na = eos(fig, plotter)
    _, nb = liquid(fig, plotter)
    _, nc = conductivity(fig, plotter)
    _, nd = adsorption(fig, plotter)
    _, ne = efficiency(fig, plotter)
    _, nf = exploration(fig, plotter)
    for centre, top in [(49.5, 134), (142.5, 130.5)]:
        legend(fig, [Patch(facecolor=RED, label='Teacher'), Patch(facecolor=BLUE, label='Student')], centre, top, 2)
    for label, x, top in [('a',2,1.5), ('b',95,1.5), ('c',2,66.5), ('d',95,66.5), ('e',2,126.5), ('f',95,126.5)]:
        plotter.label(fig, label, x, top)
    plotter.save(fig, 'Fig6', list('abcdef'), dict(EOS_points=na, scattering_points=nb,
        conductivity=nc, adsorption_points=nd, efficiency=ne, exploration_costs=nf))
