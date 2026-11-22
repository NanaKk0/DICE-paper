"""Current paper panels from archived numerical observations (Python)."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import font_manager
from PIL import Image
import numpy as np
from property_panels import PaperPlotter as PhysicalPlotter, configure

COLOURS=['#E53935','#E69F00','#009EAF','#9162D6','#303030']
POOLS=['gamma506','liquid423']
POOL_NAMES=[r'$\gamma$-Smrčok','Liquid–amorphous']
ROUTES=['T0_70','Hybrid_70','TR_70','DFT_20','DFT_70']
ROUTE_NAMES=[r'$T_0$','Hybrid',r'$T_R$','DFT-20','DFT-70']


class PaperPlotter(PhysicalPlotter):
    def main2(self):
        fig=self.start(100)
        tails=self.csv('data/controlled/Fig2a_paired_tails.csv')
        t=self.csv('data/controlled/Fig2b_teacher_pool_metrics.csv')
        for i,pool in enumerate(POOLS):
            ax=self.axis(fig,19,15+43*i,66,26)
            for route,label,colour in zip(ROUTES[:-1],ROUTE_NAMES[:-1],COLOURS):
                g=tails[(tails.pool==pool)&(tails.route==route)]
                s=g.groupby('q_percent').difference_eV_A.mean()
                ax.plot(s.index,s,color=colour,label=label,lw=1)
            ax.axhline(0,color='.4',lw=.6)
            ax.set(xscale='log',xlim=(.01,1),ylabel='Tail error − DFT-70\n(eV Å$^{-1}$)')
            ax.set_xticks([.01,.1,1],['0.01','0.1','1'])
            ax.text(.98,.93,POOL_NAMES[i],ha='right',va='top',transform=ax.transAxes)
            if i==1:ax.set_xlabel('Top atom–frame fraction (%)')
        self.legend(fig,[Line2D([],[],color=c,label=n) for c,n in zip(COLOURS,ROUTE_NAMES[:-1])],center=.29,y=.97,ncol=4)
        ax=self.axis(fig,114,29,61,46)
        teachers=['T0','TR_gamma','TR_liquid']
        for i,pool in enumerate(POOLS):
            d=t[t.evaluation_pool==pool]
            s=d.groupby('teacher').force_component_RMSE_eV_A.agg(['mean','std']).reindex(teachers)
            ax.bar(np.arange(3)+(i-.5)*.34,s['mean'],.34,color=COLOURS[2*i],label=POOL_NAMES[i],
                   yerr=s['std'].fillna(0),capsize=2)
            for j,name in enumerate(teachers):
                values=d.loc[d.teacher==name,'force_component_RMSE_eV_A'].to_numpy()
                ax.scatter(j+(i-.5)*.34+np.linspace(-.055,.055,len(values)),values,
                           s=7,facecolor='white',edgecolor=COLOURS[2*i],lw=.5,zorder=4)
        ax.set_xticks(range(3),[r'$T_0$',r'$T_R$ + γ',r'$T_R$ + liquid'])
        ax.set_ylabel('Force-component RMSE (eV Å$^{-1}$)')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.42),ncol=1)
        self.label(fig,'a',2,4);self.label(fig,'b',97,4)
        self.save(fig,'Fig2',list('ab'),dict(training_geometries=70,held_out_frames=[446,363],paired_repeats=5))

    def main3(self):
        fig=self.start(166)
        data=self.csv('data/routing/Fig3_scatter_source.csv')
        norm=LogNorm(.005,.3)
        cmap=LinearSegmentedColormap.from_list('error',['#218BC3','#E1E1E1','#F33131'])
        for col,(pool,name) in enumerate(zip(['gamma','liquid'],POOL_NAMES)):
            x=16+61*col
            for row,method,panel,teacher in [(0,'QBC','ab'[col],'T0'),(1,'GMM','de'[col],'T0'),(2,'GMM','de'[col],'T1')]:
                top=[24,78,121][row]
                ax=self.axis(fig,x,top,43,30)
                d=data[(data.panel==panel)&(data.teacher==teacher)]
                err=d.teacher_force_component_RMSE_eV_A
                mark=ax.scatter(d.x,err,c=err,cmap=cmap,norm=norm,s=5,alpha=.8,lw=0)
                for value in ([.03,.05,.08] if method=='QBC' else [.1]):ax.axvline(value,color='.3',ls='--' if method=='QBC' else ':',lw=.7,zorder=8)
                ax.axhline(.05,color='.5',lw=.7,zorder=8)
                ax.set(xscale='log',ylim=(0,.36))
                if method=='QBC':
                    ax.set(xlim=(.008,.45),xlabel='Maximum atomic force\ndisagreement, σ (eV Å$^{-1}$)')
                    ax.set_xticks([.01,.05,.1],['0.01','0.05','0.1'])
                else:
                    ax.set_xlim(.043,1.02);ax.set_xticks([.05,.1,.2,.5,1],['0.05','0.1','0.2','0.5','1'])
                    if row==2:ax.set_xlabel('Support score, $p$')
                if col==0:ax.set_ylabel('Force-component RMSE\n(eV Å$^{-1}$)')
                ax.text(.96,.91,r'$T_0$' if teacher=='T0' else r'$T_1$',transform=ax.transAxes,ha='right')
                if row<2:ax.set_title(method+', '+name,pad=7)
            self.label(fig,'ab'[col],x-14,16);self.label(fig,'de'[col],x-14,69)
        ca=fig.add_axes([.20,.94,.42,.013]);cb=fig.colorbar(mark,cax=ca,orientation='horizontal')
        cb.set_ticks([.005,.01,.02,.05,.1,.3]);cb.set_ticklabels(['0.005','0.01','0.02','0.05','0.1','0.3'])
        ca.set_title('Teacher force-component RMSE (eV Å$^{-1}$; log scale)',pad=5)
        pairs=self.csv('data/routing/Fig3g_relative_P99_paired_source.csv')
        ax=self.axis(fig,145,24,30,127)
        for block,(pool,name) in enumerate(zip(POOLS,POOL_NAMES)):
            subset=pairs[pairs.pool==pool]
            groups=[('QBC',t,'T0') for t in ['0.03','0.05','0.08']]+[('GMM','$p>0.1$',t) for t in ['T0','T1']]
            for j,(router,criterion,teacher) in enumerate(groups):
                d=subset[(subset.router==router)&(subset.criterion.astype(str)==criterion)&(subset.teacher_label_source==teacher)]
                if len(d)!=5:raise ValueError('Five paired student observations required')
                v=d.relative_P99_difference_percent; y=block*7+j
                ax.barh(y,v.mean(),height=.3,color=COLOURS[0 if block else 2] if router=='QBC' else COLOURS[3])
                ax.errorbar(v.mean(),y,xerr=v.std(ddof=1),fmt='none',ecolor='.25',capsize=2,lw=.8)
            ax.text(5,block*7-1.1,name,ha='center')
            ax.text(-.39,block*7+1,'QBC\nσ (eV Å$^{-1}$)',transform=ax.get_yaxis_transform(),ha='center',va='center',rotation=90)
            ax.text(-.39,block*7+3.5,'GMM\n$p>0.1$',transform=ax.get_yaxis_transform(),ha='center',va='center',rotation=90)
        ax.axvline(0,color='.4',ls='--',lw=.7)
        ax.set(ylim=(11.8,-2),xlim=(-10,20),xlabel='Relative force-vector\nerror P99 difference (%)')
        ax.set_yticks([0,1,2,3,4,7,8,9,10,11],['0.03','0.05','0.08',r'$T_0$',r'$T_1$']*2)
        ax.set_title('Student accuracy',pad=7)
        self.label(fig,'c',130,16)
        self.save(fig,'Fig3',list('abcde'),dict(error_threshold_eV_A=.05,QBC_thresholds=[.03,.05,.08],support_threshold=.1,relative_P99='mean of per-repeat percentage differences'))

    def main4(self):
        fig=self.start(198)
        groups=['teacher_140','student_before_940','added_380','test_120']
        names=['Teacher training set','Student training set','Added configurations','Test set']
        colours=['#E53935','#E69F00','#3B70B2','#009EAF']
        spectra=self.csv('data/structural_analysis/MEAN_STRUCTURE_FACTORS.csv')
        metrics=self.csv('data/structural_analysis/ORDER_METRICS.csv')
        a=self.axis(fig,17,17,68,31)
        for group,colour,ls in zip(groups,colours,['-','--','--','-']):
            d=spectra[spectra.group==group]
            a.plot(d.q_Ainv,d.mean_S_OO,color=colour,ls=ls,lw=1,zorder=3 if ls=='--' else 2)
        a.set(xlim=(1,10),xlabel='Scattering vector, $q$ (Å$^{-1}$)',ylabel=r'$S_{\mathrm{OO}}(q)$')
        self.legend(fig,[Line2D([],[],color=c,ls=l,label=n) for c,l,n in zip(colours,['-','--','--','-'],names)],center=.50,y=.99,ncol=2)
        for slot,i in enumerate([0,3,1,2]):
            col,row=slot%2,slot//2
            ax=self.axis(fig,111+35*col,17+18*row,28,12)
            ax.scatter(metrics.O_qbar6_mean,metrics.O_reciprocal_max_coherence,s=1,c='.8',alpha=.25,lw=0)
            d=metrics[metrics.group==groups[i]]
            ax.scatter(d.O_qbar6_mean,d.O_reciprocal_max_coherence,s=2,c=colours[i],lw=0)
            ax.set(xlim=(0,.6),ylim=(0,1.04));ax.set_xticks([0,.3,.6]);ax.set_yticks([0,.5,1])
            ax.tick_params(labelsize=5.5,labelbottom=row==1,labelleft=col==0)
        fig.text(.81,.721,r'Bond-orientational order, $\langle\bar{q}_6\rangle$',ha='center',size=6)
        fig.text(.55,.84,'Normalized peak intensity',rotation=90,ha='center',va='center',size=6)
        teachers=self.csv('data/crossphase/TEACHER_PLOTTED_DATA.csv')
        c=self.axis(fig,17,68,68,24)
        c.bar([0,1],teachers.force_component_RMSE_eV_A,color=[colours[0],colours[3]],width=.6)
        c.set_xticks([0,1],['Pretrained\nteacher','Fine-tuned\nteacher']);c.set_ylabel('Force-component RMSE\n(eV Å$^{-1}$)')
        progress=self.csv('data/crossphase/STUDENT_PLOTTED_DATA.csv')
        d=self.axis(fig,110,68,65,24);d.axvspan(2.5,10.5,color='.92')
        d.plot(progress['round'],progress.force_component_RMSE_eV_A,'o-',color=colours[3],ms=2.5)
        d.set(xlabel='Cycle index',ylabel='Force-component RMSE\n(eV Å$^{-1}$)')
        d.set_xticks([0,3,10,20,27],['r0','r3','r10','r20','r27'])
        cmap=LinearSegmentedColormap.from_list('mixing',matplotlib.colormaps['viridis'](np.linspace(.28,1,256)))
        for j,element in enumerate(['Al','O']):
            ax=self.axis(fig,16+88*j,116,69,35)
            t=self.csv(f'data/crossphase/TSNE_{element}.csv')
            z=ax.scatter(t.tsne_1,t.tsne_2,c=t.local_mixing,cmap=cmap,norm=PowerNorm(.5,0,1),s=2,lw=0,rasterized=True)
            ax.set(xticks=[],yticks=[]);ax.set_title(element+' atomic environments',pad=4)
            for spine in ax.spines.values():spine.set_visible(False)
        cb=fig.colorbar(z,cax=fig.add_axes([.34,.19,.32,.012]),orientation='horizontal');cb.set_label('Neighbourhood mixing')
        for i,cn in enumerate([4,5,6]):
            ax=self.axis(fig,9+58*i,167,53,26)
            source=f'data/local_environments/AlO{cn}_polyhedra.png';self.sources.add(source)
            ax.imshow(Image.open(self.a.path(source)));ax.axis('off');ax.set_title(f'AlO$_{cn}$',pad=2)
        for label,x,y in [('a',2,10),('b',95,10),('c',2,60),('d',94,60),('e',2,105),('f',2,158)]:self.label(fig,label,x,y)
        self.save(fig,'Fig4',list('abcdef'),dict(groups=groups,all_projection_points=True,mixing_radius=8))

    def main5(self):
        # The production observations and physical calculations are shared with
        # the preceding archive; only the current high-error bar replaces the
        # earlier transition decomposition.
        old=self.save
        def capture(fig,stem,panels,selections=None):
            ax=fig.axes[3];ax.clear()
            atoms=self.csv('data/ysz/validation_atoms.csv.gz')
            force=self.csv('data/accuracy/S_direct_YSZ500_parity_force.csv')
            if len(force)!=3*len(atoms):raise ValueError('YSZ atomic ordering')
            reference=force.DFT_eV_A.to_numpy().reshape(-1,3)
            atoms=atoms.assign(direct_error=np.linalg.norm(force.direct_eV_A.to_numpy().reshape(-1,3)-reference,axis=1),
                               dice_error=np.linalg.norm(force.DICE_eV_A.to_numpy().reshape(-1,3)-reference,axis=1))
            for j,element in enumerate(['O','Zr','Y']):
                g=atoms[atoms.element==element]
                for k,(column,colour) in enumerate([('direct_error',COLOURS[0]),('dice_error',COLOURS[2])]):
                    ax.bar(j+(k-.5)*.32,100*(g[column]>.3).mean(),.32,color=colour)
            ax.set(xticks=range(3),xticklabels=['O','Zr','Y'],ylabel='High-error observations (%)')
            old(fig,'Fig5',panels,selections)
        self.save=capture
        try:super().main4()
        finally:self.save=old

    def main6(self):
        old=self.save
        def capture(fig,stem,panels,selections=None):
            adsorption,conductivity=fig.axes[3],fig.axes[4]
            left,right=adsorption.get_position().frozen(),conductivity.get_position().frozen()
            adsorption.set_position(right);conductivity.set_position(left)
            # The two scattering axes share b; transport is c and adsorption d.
            for t in list(fig.texts):
                if t.get_text() in list('abcde'):t.remove()
            for label,x,y in [('a',2,5),('b',94,5),('c',2,78),('d',94,78)]:self.label(fig,label,x,y)
            old(fig,'Fig6',list('abcd'),selections)
        self.save=capture
        try:super().main5()
        finally:self.save=old

    def main7(self):
        super().main6()

    def supplement(self,n):
        stem='SupplementaryFig'+str(n)
        if n in [1,2]:
            fig=self.start(115)
            for j,pool in enumerate(POOLS):
                if n==1:
                    teacher=self.csv('data/controlled/teacher_metrics_per_repeat.csv')
                    ax=self.axis(fig,18+90*j,15,65,32)
                    own=teacher[teacher.pool==pool]
                    for k,name in enumerate(['T0','TR']):
                        d=own[own.teacher==name].groupby('scope').force_component_RMSE_eV_A.agg(['mean','std']).reindex(['all70','unqueried50','heldout_pool'])
                        ax.bar(np.arange(3)+(k-.5)*.32,d['mean'],.32,yerr=d['std'],capsize=2,color=COLOURS[k*2],label=r'$T_0$' if k==0 else r'$T_R$')
                    ax.set_xticks(range(3),['Training 70','Unqueried 50','Held out']);ax.set_ylabel('Force-component RMSE (eV Å$^{-1}$)');ax.legend()
                    raw=self.csv('data/controlled/metrics_per_repeat.csv');raw=raw[(raw.pool==pool)&(raw.scope=='excluded_training_frames')]
                    ax2=self.axis(fig,18+90*j,76,65,26)
                    stats=raw.groupby('route').force_component_RMSE_eV_A.agg(['mean','std']).reindex(ROUTES)
                    ax2.bar(range(5),stats['mean'],yerr=stats['std'],capsize=2,color=COLOURS)
                    ax2.set_xticks(range(5),ROUTE_NAMES);ax2.set_ylabel('Student RMSE (eV Å$^{-1}$)')
                else:
                    accept=self.csv('data/si_comparisons/routing_acceptance_per_repeat.csv');own=accept[accept.pool==pool]
                    ax=self.axis(fig,18+90*j,15,65,32)
                    groups=['QBC_0.03','QBC_0.05','QBC_0.08','GMM_T0','GMM_T1']
                    stats=own.groupby('group').high_error_fraction_percent.agg(['mean','std']).reindex(groups)
                    ax.bar(range(5),stats['mean'],yerr=stats['std'].fillna(0),capsize=2,color=[*([COLOURS[0]]*3),*([COLOURS[3]]*2)])
                    ax.set_xticks(range(5),['0.03','0.05','0.08',r'$T_0$',r'$T_1$']);ax.set_ylabel('Accepted-set high-error fraction (%)')
                    raw=self.csv('data/si_comparisons/routing_RMSE_paired_per_repeat.csv');own=raw[raw.pool==pool]
                    stats=own.groupby('group').difference_eV_A.agg(['mean','std']).reindex(groups)
                    ax2=self.axis(fig,18+90*j,76,65,26)
                    ax2.bar(range(5),stats['mean'],yerr=stats['std'],capsize=2,color=[*([COLOURS[0]]*3),*([COLOURS[3]]*2)])
                    ax2.axhline(0,color='.4',lw=.6);ax2.set_xticks(range(5),['0.03','0.05','0.08',r'$T_0$',r'$T_1$']);ax2.set_ylabel('RMSE difference (eV Å$^{-1}$)')
                ax.set_title(POOL_NAMES[j],pad=7)
            for label,x,y in [('a',2,4),('b',92,4),('c',2,63),('d',92,63)]:self.label(fig,label,x,y)
            self.save(fig,stem,list('abcd'));return
        if n in [3,4,5]:
            system={3:'Al',4:'YSZ',5:'MIL'}[n]
            self.grid(stem,[lambda ax,q=q,r=r:self.parity_panel(ax,system,q,r) for q in ['energy','force'] for r in ['Direct','DICE']],height=145,handles=self.model_handles(['Direct','DICE']))
        elif n==6:self.grid(stem,[lambda ax,p=p:self.mil_eos_panel(ax,p) for p in 'ABCDEF'],cols=3,height=112,handles=self.model_handles())
        elif n in [7,13]:
            paths=list(self.csv('data/physical_validation/MIL_barriers.csv' if n==13 else 'data/prediction_and_mil/all_path_profiles.csv').path.unique())
            def draw(ax,path):
                self.profile(ax,path,relaxed=n==13);ax.text(.03,.96,path.replace('_Ecell',' (alternate cell)'),transform=ax.transAxes,va='top',size=6)
            self.grid(stem,[lambda ax,p=p:draw(ax,p) for p in paths],cols=3,height=165,handles=self.model_handles(['DFT','DICE'] if n==13 else ['DFT','Direct','DICE']))
        elif n==8:self.grid(stem,[lambda ax:self.phase_panel(ax,'relative_energy_eV_per_fu'),lambda ax:self.phase_panel(ax,'volume_A3_per_fu')],height=110,handles=self.model_handles(['DFT','DICE']))
        elif n==9:
            self.grid(stem,[lambda ax,p=p:self.dispersion(ax,p) for p in ['alpha','kappa','theta']],cols=1,height=165,handles=self.model_handles(['DFT','DICE']))
        elif n==10:self.grid(stem,[self.velocities,self.coexistence_trace],height=85)
        elif n==11:self.grid(stem,[self.normalized],cols=1,height=80)
        elif n==12:self.grid(stem,[self.thermal],cols=1,height=88)
        else:raise ValueError(n)


def rebuild(archive,output,selected=None):
    configure();p=PaperPlotter(archive,output)
    figures=archive.registry['figures']
    if selected:
        unknown=set(selected)-{f['id'] for f in figures}
        if unknown:raise ValueError('Unknown figure IDs: '+str(unknown))
        figures=[f for f in figures if f['id'] in selected]
    for f in figures:
        kind,number=f['id'].split(':');n=int(number)
        if f['id']=='main:1':
            import cairosvg
            source=archive.path(f['artwork']);stem=output/f['stem']
            shutil.copyfile(source,stem.with_suffix('.svg'))
            cairosvg.svg2pdf(url=str(source),write_to=str(stem.with_suffix('.pdf')))
            cairosvg.svg2png(url=str(source),write_to=str(stem.with_suffix('.png')),dpi=300)
            p.records.append(dict(stem=f['stem'],sources=[f['artwork']],role='approved editable schematic'))
        elif kind=='main':getattr(p,'main'+number)()
        else:p.supplement(n)
        print('Built '+f['id'],flush=True)
    (output/'paper-figures.json').write_text(json.dumps(dict(backend='python',font='Arial',snapshot=archive.registry['snapshot'],figures=p.records),indent=2)+'\n')
    return dict(figures=len(p.records),report='paper-figures.json')
