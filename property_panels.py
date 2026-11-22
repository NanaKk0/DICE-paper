"""Reconstruct material-property and model-comparison panels from source data."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

from property_data_plots import Plotter

COLORS = ["#009EAF", "#E53935", "#13AA7A", "#9162D6", "#E7A817", "#2377D8"]
MODEL = {"DFT": ("#303030", "--", "o"),
         "Direct": (COLORS[1], ":", "^"), "DICE": (COLORS[0], "-", "s")}
NAMES = {"DFT": "DFT", "Direct": "Directly fine-tuned student", "DICE": "DICE-distilled student"}
PHASES = {"alpha": r"$\alpha$", "kappa": r"$\kappa$", "theta": r"$\theta$",
          "delta_jp50s1": r"$\delta$-JP50s1", "gamma_digne": r"$\gamma$-Digne"}


def configure():
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "lines.linewidth": 1.,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial", "mathtext.sf": "Arial"})


class PaperPlotter(Plotter):
    def start(self, height=130):
        self.sources = set()
        return plt.figure(figsize=(180/25.4, height/25.4))

    @staticmethod
    def axis(fig, x, top, width, height):
        w, h = fig.get_size_inches()*25.4
        return fig.add_axes([x/w, 1-(top+height)/h, width/w, height/h])

    @staticmethod
    def label(fig, text, x, top):
        w, h = fig.get_size_inches()*25.4
        fig.text(x/w, 1-top/h, text, va="top", ha="left", weight="bold", size=10)

    @staticmethod
    def legend(fig, handles, center=.54, y=.99, ncol=None):
        return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(center, y),
                          ncol=ncol or len(handles), borderaxespad=0, columnspacing=1.2,
                          handlelength=1.9, handletextpad=.4)

    @staticmethod
    def model_handles(keys=("DFT", "Direct", "DICE")):
        return [Line2D([], [], color=MODEL[k][0], ls=MODEL[k][1],
                       marker=MODEL[k][2], ms=3, label=NAMES[k]) for k in keys]

    def save(self, fig, stem, panels, selections=None):
        # The source data and final manuscript artwork stay read-only.
        for ax in fig.axes:
            if not ax.axison: continue
            if ax.get_xscale()=="log": ax.xaxis.set_minor_locator(NullLocator())
            if ax.get_yscale()=="log": ax.yaxis.set_minor_locator(NullLocator())
            for getter, setter, bounds in [(ax.get_xticks, ax.set_xticks, ax.get_xlim()),
                                            (ax.get_yticks, ax.set_yticks, ax.get_ylim())]:
                lo,hi=sorted(bounds)
                ticks=getter();visible=[v for v in ticks if lo-1e-10<=v<=hi+1e-10]
                if len(visible)!=len(ticks): setter(visible)
        for text in fig.findobj(matplotlib.text.Text):
            text.set_text(text.get_text().replace(r"\n", "\n").replace("⁻¹", r"$^{-1}$").replace("³", r"$^3$"))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside=[]
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_visible() and t.get_text().strip():
                b=t.get_window_extent(renderer)
                if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1:
                    outside.append(t.get_text())
        if outside:
            plt.close(fig)
            raise ValueError(f"{stem}: text outside canvas: {outside}")
        for ext in ("svg", "pdf", "png"):
            fig.savefig(self.output/(stem+"."+ext), dpi=300)
        self.records.append(dict(stem=stem, panels=panels, sources=sorted(self.sources),
            size_mm=(fig.get_size_inches()*25.4).tolist(),
            role="source-data reconstruction of paper panels; not pixel-identical artwork",
            selections=selections or {}, outside_text=[]))
        plt.close(fig)






    def dispersion(self,ax,phase="alpha",matched=False):
        filename="data/phonons/S26_phonon_dispersion.csv" if matched else "data/physical_validation/Al_phonon_band.csv"
        metadata="data/phonons/S26_q_path.json" if matched else "data/physical_validation/Al_phonon_path_metadata.json"
        d=self.csv(filename)
        if not matched:d=d[d.phase==phase]
        meta=self.a.json(metadata)["phases"][phase];self.sources.add(metadata)
        q=d[["q_index","distance"]].drop_duplicates().sort_values("q_index")
        if matched:
            # Keep the first main-path endpoint, not the duplicated L/other branch start.
            end=meta["tick_distances"][7];q=q[q.distance<=end+1e-10]
            duplicate=q[q.distance.duplicated()].q_index
            if len(duplicate):q=q[q.q_index<duplicate.iloc[0]]
            series=[("DFT","reference_frequency_THz","reference_mode"),("Direct","DFT_only_frequency_THz","reference_mode"),("DICE","DICE_frequency_THz","reference_mode")]
            ticks=meta["tick_distances"][:8];labels=meta["tick_labels"][:8]
        else:
            series=[("DFT","reference_frequency_THz","reference_mode"),("DICE","model_frequency_THz","model_mode")]
            ticks=meta["tick_distances"];labels=meta["tick_labels"]
        x=q.distance.to_numpy();breaks=np.flatnonzero(np.diff(x)<1e-10)+1
        gap=0 if matched else x.max()*.035;offset=np.zeros(len(x))
        for i in breaks:offset[i:]+=gap
        cuts=np.r_[0,breaks,len(x)]
        for model,column,mode in series:
            values=d.pivot(index="q_index",columns=mode,values=column).loc[q.q_index].to_numpy()
            for lo,hi in zip(cuts[:-1],cuts[1:]):ax.plot((x+offset)[lo:hi],values[lo:hi],color=MODEL[model][0],ls=MODEL[model][1],lw=.55)
        xp=[];shift=0;last=None
        for v in ticks:
            if last is not None and abs(v-last)<1e-10:shift+=gap
            xp.append(v+shift);last=v
        labels=[{"G":r"$\Gamma$","M1":r"$M_1$","H1":r"$H_1$"}.get(v,v) for v in labels]
        ax.set_xticks(xp,labels)
        for v in xp:ax.axvline(v,color=".88",lw=.4,zorder=0)
        ax.axhline(0,color=".6",lw=.4);ax.margins(x=0,y=.03)
        ax.set(xlabel="Wave vector",ylabel="Frequency (THz)")

    def profile(self,ax,path,relaxed=False):
        filename="data/physical_validation/MIL_relaxed_NEB_profiles.csv" if relaxed else "data/prediction_and_mil/all_path_profiles.csv"
        d=self.csv(filename);d=d[d.path==path]
        if relaxed:
            for method,model in [("DFT","DFT"),("Relaxed NEB","DICE")]:
                g=d[d.method==method].sort_values("image");c,ls,m=MODEL[model]
                ax.plot(g.normalised_arc,g.relative_energy_meV,color=c,ls=ls,marker=m,ms=2.5)
            ax.set_xlabel("Normalised path length")
        else:
            for model in ["DFT","Direct","DICE"]:
                g=d[d.route==("DICE" if model=="DFT" else model)].sort_values("image");c,ls,m=MODEL[model]
                ax.plot(g.normalized_image_index,g.DFT_relative_meV if model=="DFT" else g.model_relative_meV,color=c,ls=ls,marker=m,ms=2.5)
            ax.set_xlabel("Normalised image index")
        ax.set_ylabel("Relative energy (meV per P40)");ax.margins(y=.12)

    def main4(self):
        fig=self.start(170);frames=self.csv("data/prediction_and_mil/frames.csv")
        for j,(system,title) in enumerate([("Al2O3",r"Al$_2$O$_3$ (200)"),("MIL-120",r"MIL-120(Al) (300)"),("YSZ","YSZ (330)")]):
            ax=self.axis(fig,18+58*j,15,43,31)
            series=[frames[(frames.system==system)&(frames.route==r)].frame_RMSE.to_numpy() for r in ["Direct","DICE"]]
            if any(len(v)==0 for v in series):raise ValueError("Missing production frame errors: "+system)
            violin=ax.violinplot(series,[0,1],widths=.8,showextrema=False)
            for i,body in enumerate(violin["bodies"]):body.set_facecolor(MODEL[["Direct","DICE"][i]][0]);body.set_edgecolor(MODEL[["Direct","DICE"][i]][0]);body.set_alpha(.24)
            for i,v in enumerate(series):
                ax.scatter(np.full(len(v),i),v,s=2,c=MODEL[["Direct","DICE"][i]][0],alpha=.35,lw=0)
                ax.vlines(i,*np.percentile(v,[25,75]),color=MODEL[["Direct","DICE"][i]][0],lw=2)
                ax.plot(i,np.median(v),"o",ms=3,mfc="white",mec=MODEL[["Direct","DICE"][i]][0])
            ax.set_xticks([0,1],["Directly fine-\ntuned student","DICE-distilled\nstudent"])
            ax.text(.5,.96,title,transform=ax.transAxes,ha="center",va="top",size=7)
            ax.set_ylim(bottom=0)
            if j==0:ax.set_ylabel(r"Frame force RMSE\n(eV $\mathrm{\AA}^{-1}$)")
        b=self.axis(fig,18,70,43,35);c=self.axis(fig,85,70,91,35)
        d=self.axis(fig,18,125,66,27);e=self.axis(fig,112,125,64,27)
        transitions=self.csv("data/provenance/YSZ500k_error_transitions.csv");transitions=transitions[transitions.threshold_eV_A==.3]
        for k,transition in enumerate(["removed","introduced"]):
            g=transitions[transitions.transition==transition].set_index("element").loc[["O","Zr","Y"]]
            b.bar(np.arange(3)+(k-.5)*.32,100*g.fraction,.32,color=COLORS[k],label=transition.capitalize(),
                yerr=100*np.array([g.fraction-g.CI_low,g.CI_high-g.fraction]),capsize=2)
        b.set(xticks=range(3),xticklabels=["O","Zr","Y"],ylabel="Atom observations (%)")
        b.legend(loc="upper right",fontsize=6)
        self.dispersion(c,matched=True);self.profile(d,"C-B")
        phase=self.csv("data/mil120/mil_phase_order.csv")
        for model,g in phase.groupby("model",sort=False):
            role={"DFT":"DFT", "final/3":"DICE", "direct-500k":"Direct"}[model]
            color,ls,marker=MODEL[role];e.plot(g.phase,g.relative_energy_meV_cell,color=color,ls=ls,marker=marker,ms=3)
        e.set(xlabel="Framework configuration",ylabel="Relative energy (meV per P40)")
        self.legend(fig,self.model_handles())
        for label,x,y in [("a",2,4),("b",2,61),("c",69,61),("d",2,116),("e",96,116)]:self.label(fig,label,x,y)
        self.save(fig,"Fig5",list("abcde"),dict(frame_observations=len(frames),transition_threshold_eV_A=.3,main_phonon_path="GMKGALHA",representative_path="C-B"))

    def main5(self):
        fig=self.start(140);a=self.axis(fig,19,16,64,42)
        b=self.axis(fig,110,16,65,19);c=self.axis(fig,110,48,65,19)
        d=self.axis(fig,19,88,64,40);e=self.axis(fig,110,88,65,40)
        eos=self.csv("data/physical_validation/AL_EOS_PLOT.csv")
        handles=[]
        for i,(phase,g) in enumerate(eos.groupby("phase",sort=False)):
            color=COLORS[i];g=g.sort_values("V_fu")
            a.plot(g.V_fu/5,g.MLIP_relative_meV_atom,color=color)
            a.plot(g.V_fu/5,g.DFT_relative_meV_atom,color=color,ls="--",marker="o",ms=2)
            handles.append(Line2D([],[],color=color,label=PHASES.get(phase,phase)))
        a.set(xlabel=r"Volume ($\mathrm{\AA}^3$ atom$^{-1}$)",ylabel=r"$E-E_{\alpha,\mathrm{ref}}$ (meV atom$^{-1}$)")
        self.legend(fig,handles,center=.28,y=.99,ncol=3)
        a.legend(handles=[Line2D([],[],color=".2",ls=ls,label=name) for ls,name in [("-","DICE"),("--","DFT")]],loc="lower right")
        gr=self.csv("data/physical_validation/AL_G_figure_source_data.csv")
        for ax,kind,name in [(b,"X","X-ray"),(c,"N","Neutron")]:
            g=gr[gr.kind==kind].sort_values("r_A")
            if len(g)==0:raise ValueError("Missing scattering kind "+kind)
            ax.plot(g.r_A,g.G_mean,color=COLORS[0],label="DICE")
            ax.fill_between(g.r_A,g.G_mean-g.G_sd,g.G_mean+g.G_sd,color=COLORS[0],alpha=.18,lw=0)
            ax.plot(g.r_A,g.G_experiment,color=".25",ls="--",label="Experiment")
            ax.set(xlim=(0,10),xlabel=r"$r$ ($\mathrm{\AA}$)",ylabel=r"$G(r)$")
            ax.text(.95,.9,name,transform=ax.transAxes,ha="right",va="top")
            if kind=="X":ax.legend(loc="lower right",ncol=2,fontsize=6)
        ads=self.csv("data/physical_validation/MIL_MEAN_SD.csv")
        for i,(temperature,g) in enumerate(ads.groupby("temperature_K",sort=True)):
            g=g.sort_values("pressure_bar");color=COLORS[i]
            d.errorbar(g.pressure_bar,g.uptake_mean_mmol_g,yerr=g.uptake_sd_mmol_g,color=color,fmt="o-",ms=2,capsize=1.5)
            d.plot(g.pressure_bar,g.experiment_fit_mmol_g,color=color,ls="--")
        d.set(xlabel=r"CO$_2$ pressure (bar)",ylabel=r"CO$_2$ uptake (mmol g$^{-1}$)")
        temps=[Line2D([],[],color=COLORS[i],label=f"{v} K") for i,v in enumerate(sorted(ads.temperature_K.unique()))]
        ld=d.legend(handles=temps,ncol=3,loc="lower right",bbox_to_anchor=(1,.17),fontsize=6);d.add_artist(ld)
        d.legend(handles=[Line2D([],[],color=".2",ls=ls,label=name) for ls,name in [("-","DICE"),("--","Experiment")]],ncol=2,loc="lower right",fontsize=6)
        cond=self.csv("data/physical_validation/YSZ_CONDUCTIVITY.csv")
        for i,comp in enumerate(["8YSZ","10YSZ"]):
            g=cond[cond.simulation_label==comp].sort_values("T_K");color=COLORS[i]
            e.errorbar(1000/g.T_K,g.simulation_NE_mean_S_cm,yerr=g.simulation_NE_SD_S_cm,color=color,fmt="o-",ms=3,capsize=2,label=comp+" DICE")
            experiment_composition=g.experiment_mol_percent.unique()
            if len(experiment_composition)!=1:raise ValueError("Mixed experimental compositions")
            e.plot(1000/g.T_K,g.experiment_S_cm,color=color,ls="--",label=f"{experiment_composition[0]:g}YSZ Experiment")
        e.set(xlabel=r"$1000/T$ (K$^{-1}$)",ylabel=r"$\sigma$ (S cm$^{-1}$)",yscale="log")
        e.set_yticks([.05,.1,.2,.4],["0.05","0.10","0.20","0.40"])
        e.legend(ncol=2,loc="upper right",fontsize=6)
        for label,x,y in [("a",2,5),("b",94,5),("c",94,40),("d",2,78),("e",94,78)]:self.label(fig,label,x,y)
        self.save(fig,"Fig6",list("abcde"),dict(conductivity_compositions=["8YSZ","10YSZ"],conductivity_definition="tracer Nernst–Einstein, experimental DC",scattering_offsets=False))

    def main6(self):
        fig=self.start(78);a=self.axis(fig,19,15,79,50);b=self.axis(fig,120,15,56,50)
        data=self.csv("data/performance/THREE_SYSTEM_LATENCY.csv");guides=self.csv("data/performance/OOM_DISPLAY_GUIDES.csv")
        system_names={"al2o3":r"Al$_2$O$_3$","mil":"MIL-120(Al)", "ysz":"YSZ"}
        for i,system in enumerate(["al2o3","mil","ysz"]):
            for model,style in [("Neo","--"),("Nano","-")]:
                g=data[(data.system==system)&(data.model==model)].sort_values("atoms")
                if len(g)==0:raise ValueError("Missing latency series "+system+model)
                a.plot(g.atoms,g.latency_ms_median,color=COLORS[i],ls=style,marker=["o","s","^"][i],ms=2.5)
                a.fill_between(g.atoms,g.latency_ms_q25,g.latency_ms_q75,color=COLORS[i],alpha=.1)
                z=guides[(guides.system==system)&(guides.model==model)].iloc[0]
                a.plot([z.last_measured_atoms,z.OOM_atoms_observed],[z.last_measured_latency_ms,z.display_only_endpoint_ms],color=COLORS[i],ls=":",lw=.8)
                a.plot(z.OOM_atoms_observed,z.display_only_endpoint_ms,"x",ms=4,color=COLORS[i])
        a.set(xscale="log",yscale="log",xlabel="Number of atoms",ylabel="Time per MD step (ms)")
        self.legend(fig,[Line2D([],[],color=".2",ls="--",label="DPA4-Neo (Teacher)"),Line2D([],[],color=".2",ls="-",label="DPA4C-Nano (Student)")],y=.99)
        a.legend(handles=[Line2D([],[],color=COLORS[i],marker=["o","s","^"][i],label=system_names[s]) for i,s in enumerate(system_names)],loc="lower right")
        cost=self.csv("data/sampling_costs/DEVELOPMENT_SAMPLING_COST.csv").set_index('system').loc[['Al2O3/AL3','YSZ/AL2','MOF-MIL-120']].reset_index()
        for i,row in cost.iterrows():
            b.bar(i-.16,row.student_recorded_h,.32,color=COLORS[0]);b.bar(i+.16,row.teacher_estimated_h,.32,color=COLORS[1])
            b.text(i,row.teacher_estimated_h+12,f"{row.teacher_student_ratio:.1f}×",ha="center",size=7)
        b.set(xticks=range(3),xticklabels=[r"Al$_2$O$_3$","YSZ","MIL-120(Al)"],ylabel="Sampling time (h)",ylim=(0,350))
        b.legend(handles=[Patch(color=COLORS[0],label="Student (recorded)"),Patch(color=COLORS[1],label="Teacher (estimated)")],loc="upper right",fontsize=6)
        self.label(fig,"a",2,3);self.label(fig,"b",104,3)
        self.save(fig,"Fig7",["a","b"],dict(oom="observed atom count; saved display-only dashed extension, not measured latency",costs="startup-inclusive native-rate estimates"))

    def grid(self,stem,callbacks,cols=2,height=None,handles=None,share_row_y=False):
        rows=int(np.ceil(len(callbacks)/cols));height=height or min(165,rows*48+16)
        fig=self.start(height)
        top=15 if handles else 9;bottom=13;gap_y=16;gap_x=24
        width=(180-20-5-(cols-1)*gap_x)/cols
        ph=(height-top-bottom-(rows-1)*gap_y)/rows
        for i,draw in enumerate(callbacks):
            row,col=divmod(i,cols);x=20+col*(width+gap_x);y=top+row*(ph+gap_y)
            ax=self.axis(fig,x,y,width,ph);draw(ax);self.label(fig,chr(97+i),x-17,y-7)
        if share_row_y:
            for start in range(0,len(fig.axes),cols):
                group=fig.axes[start:start+cols]
                limits=[ax.get_ylim() for ax in group]
                common=(min(v[0] for v in limits),max(v[1] for v in limits))
                for ax in group:ax.set_ylim(common)
        if handles:self.legend(fig,handles,ncol=min(len(handles),3))
        self.save(fig,stem,[chr(97+i) for i in range(len(callbacks))])



    def parity_panel(self,ax,system,quantity,role):
        series={name:(x,y) for name,x,y in self.parity_data(system,quantity)}
        x,y=series[role];lo=min(min(v.min(),w.min()) for v,w in series.values());hi=max(max(v.max(),w.max()) for v,w in series.values());pad=(hi-lo)*.04
        color=MODEL[role][0]
        ax.scatter(x,y,s=4 if quantity=="energy" else .35,alpha=.85 if quantity=="energy" else .22,c=color,lw=0,marker="o",rasterized=quantity=="force")
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],ls="--",color=".4",lw=.6)
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),aspect="equal")
        unit=r"meV atom$^{-1}$" if quantity=="energy" else r"eV $\mathrm{\AA}^{-1}$"
        ax.set(xlabel=f"DFT ({unit})",ylabel=f"Predicted ({unit})")
        ax.text(.04,.95,f"RMSE = {np.sqrt(np.mean((y-x)**2)):.4f}",transform=ax.transAxes,va="top",size=6)


    def mil_eos_panel(self,ax,phase):
        d=self.csv("data/prediction_and_mil/all_EOS.csv");d=d[d.phase==phase]
        for role in ["DFT","Direct","DICE"]:
            g=d[d.route==("DICE" if role=="DFT" else role)].sort_values("volume_A3")
            c,ls,m=MODEL[role];ax.plot(g.volume_A3,g.dft_relative_meV if role=="DFT" else g.sp_relative_meV,color=c,ls=ls,marker=m,ms=2.5)
        ax.set(xlabel=r"Volume ($\mathrm{\AA}^3$)",ylabel="Energy (meV per P40)")
        ax.text(.96,.96,phase,transform=ax.transAxes,ha="right",va="top")

    def phase_panel(self,ax,field):
        d=self.csv("data/physical_validation/Al_eleven_phase_comparison.csv")
        factor=1000 if field=="relative_energy_eV_per_fu" else 1;y=np.arange(len(d))
        for role,offset in [("DFT",-.1),("DICE",.1)]:
            ax.scatter(d[field+"_"+role]*factor,y+offset,c=MODEL[role][0],s=14,marker=MODEL[role][2])
        ax.set_yticks(y,[PHASES.get(v,v.replace("_","-")) for v in d.phase]);ax.invert_yaxis()
        ax.set_xlabel("Relative energy (meV / Al$_2$O$_3$)" if factor==1000 else r"Volume ($\mathrm{\AA}^3$ / Al$_2$O$_3$)")

    def coexistence_trace(self,ax):
        # Original Note7 loop: all 3 temperatures × 3 velocity repeats.
        for i,temp in enumerate([2330,2340,2350]):
            name="AL_INTERFACE_2350.csv" if temp==2350 else "AL_INTERFACE_2330_2340.csv"
            d=self.csv("data/physical_validation/"+name);d=d[d.temperature_K==temp]
            for repeat,g in d.groupby("replica",sort=True):
                # Source is the archived interface proxy; do not derive a new one.
                column="solid_fraction" if "solid_fraction" in g else "solid_fraction_proxy"
                ax.plot(g.time_ps,g[column],color=COLORS[i],ls=["-","--",":"][int(repeat)-1],lw=.6)
        ax.set(xlabel="Time (ps)",ylabel="Solid-fraction proxy")

    def velocities(self,ax):
        d=self.csv('data/physical_validation/AL_COEXISTENCE.csv')
        ax.errorbar(d.temperature_K,d.velocity_m_s,
                    yerr=[d.velocity_m_s-d.ci95_low,d.ci95_high-d.velocity_m_s],fmt='o',ms=3)
        ax.axhline(0,color='black',lw=.6)
        ax.set(xlabel='Temperature (K)',ylabel='Interface velocity (m s$^{-1}$), 95% interval')
        ax.set_xticks([2330,2340,2350],["2330","2340","2350"])

    def normalized(self,ax):
        d=self.csv("data/ysz/transport_normalized_whole_preparation_bootstrap.csv");d=d[d.metric=="D_star"]
        comps=["5p8YSZ","8YSZ","10YSZ","12YSZ","14p3YSZ"]
        for i,(temp,g) in enumerate(d.groupby("T",sort=True)):
            g=g.set_index("comp").loc[comps]
            ax.errorbar([5.8,8,10,12,14.3],g.ratio,yerr=[g.ratio-g.CI_low,g.CI_high-g.ratio],color=COLORS[i],fmt=["o-","s--","^:","D-.","v-"][i],ms=3,capsize=2,label=f"{temp} K")
        ax.set(xlabel=r"Y$_2$O$_3$ content (mol%)",ylabel=r"$D^*_{\mathrm{O}}(x,T)/D^*_{\mathrm{O}}(5.8,T)$")
        ax.legend(ncol=3,loc="lower left")

    def thermal(self,ax):
        d=self.csv("data/physical_validation/YSZ_lattice_bins_common_window.csv")
        coef=self.csv("data/physical_validation/YSZ_thermal_coefficients_per_preparation.csv")
        for i,comp in enumerate(["8YSZ","10YSZ"]):
            g=d[d.composition==comp]
            for _,h in g.groupby("seed"):ax.plot(h.temperature_K,h.lattice_parameter_A,color=COLORS[i],alpha=.25,lw=.5)
            mean=g.groupby("bin_index").agg(T=("temperature_K","mean"),value=("lattice_parameter_A","mean"),sd=("lattice_parameter_A","std"))
            ax.plot(mean["T"],mean.value,color=COLORS[i],label=comp)
            ax.fill_between(mean["T"],mean.value-mean.sd,mean.value+mean.sd,color=COLORS[i],alpha=.12,lw=0)
            values=coef[coef.composition==comp].alpha_1e6_per_K
            ax.text(.04,.94-i*.12,rf"{comp}: $\alpha=({values.mean():.2f}\pm{values.std(ddof=1):.2f})\times10^{{-6}}$ K$^{{-1}}$",transform=ax.transAxes,va="top")
        ax.set(xlim=(303.15,1173.15),xlabel="Temperature (K)",ylabel=r"Equivalent cubic lattice parameter ($\mathrm{\AA}$)")
        ax.legend(loc="lower right")



