import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time, sys

import arm_ik_algebrique as ALG
import ik_numerical       as NUM

# ═══════════════════════════════════════════════════════════════
# PALETTE  Deep Space Engineering
# ═══════════════════════════════════════════════════════════════
BG       = '#080B12'
C_ALG    = '#00C896'
C_NUM    = '#4DA6FF'
C_TGT    = '#FFB800'
C_JNT    = '#2A3A4A'
C_SEP    = '#1A2332'
C_TXT    = '#4A6A8A'
C_LOG_BG = '#060D06'
C_LOG_TX = '#2A6A2A'

# ═══════════════════════════════════════════════════════════════
# CONSTANTES ROBOT
# ═══════════════════════════════════════════════════════════════
HOME   = [90, 90, 90, 0, 90]
NOMS   = ['J1  Base','J2  Epaule','J3  Coude1','J4  Coude2','J5  Pince']
LIMITS = [(0,180),(0,180),(0,180),(0,180),(0,270)]
N_ANIM = 28
L1,L2,L3,L4,L5,L6 = 66.05,41.45,82.85,82.85,73.85,90.0
OFFSET = np.array([np.pi/2,np.pi/2,np.pi/2,0,np.pi/2])
LC = L3+L4

# ═══════════════════════════════════════════════════════════════
# ÉTAT
# ═══════════════════════════════════════════════════════════════
angles_fk  = list(HOME)
angles_alg = list(HOME)
angles_num = list(HOME)
target_pt  = [None]
mode       = ['FK']
_guard     = [False]
_ik_failed = [False]   # flag hors workspace
dash = dict(st_a='--',st_n='--',err_a=None,err_n=None,t_a=None,t_n=None)
log_lines = []

# ═══════════════════════════════════════════════════════════════
# CONSOLE DEBUG
# ═══════════════════════════════════════════════════════════════
def log(msg):
    msg=msg[:32]  # tronquer pour tenir dans la colonne
    log_lines.append(msg)
    if len(log_lines) > 10: log_lines.pop(0)
    if 'log_txt' in globals():
        log_txt.set_text('\n'.join(log_lines))

class StdoutCapture:
    def write(self, msg):
        msg=msg.strip()
        if msg: log(msg)
    def flush(self): pass

# ═══════════════════════════════════════════════════════════════
# FK POINTS
# ═══════════════════════════════════════════════════════════════
def get_pts_alg(s):
    t=np.radians(s)-OFFSET; T1,T2,T3,T4,T5=t; c,s_=np.cos,np.sin
    T01=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,L1],[0,0,0,1]],dtype=float)
    T12=np.array([[-s_(T1),0,c(T1),0],[c(T1),0,s_(T1),0],[0,1,0,L2],[0,0,0,1]],dtype=float)
    T23=np.array([[-s_(T2),-c(T2),0,-L3*s_(T2)],[c(T2),-s_(T2),0,L3*c(T2)],[0,0,1,0],[0,0,0,1]],dtype=float)
    T34=np.array([[c(T3),-s_(T3),0,L4*c(T3)],[s_(T3),c(T3),0,L4*s_(T3)],[0,0,1,0],[0,0,0,1]],dtype=float)
    T45=np.array([[c(T4),0,s_(T4),0],[s_(T4),0,-c(T4),0],[0,1,0,0],[0,0,0,1]],dtype=float)
    T56=np.array([[c(T5),-s_(T5),0,0],[s_(T5),c(T5),0,0],[0,0,1,L5+L6],[0,0,0,1]],dtype=float)
    T02=T01@T12;T03=T02@T23;T04=T03@T34;T05=T04@T45;T06=T05@T56
    return np.array([[0,0,0],T01[:3,3],T02[:3,3],T03[:3,3],T04[:3,3],T05[:3,3],T06[:3,3]])

def get_pts_num(s):
    _,_,fr=NUM.fk(s)
    return np.array([[0,0,0],
        fr['T01'][:3,3],fr['T02'][:3,3],fr['T03'][:3,3],
        fr['T04'][:3,3],fr['T05'][:3,3],fr['T06'][:3,3]])

def phi_reel(s):
    t=np.radians(s)-OFFSET
    return np.degrees(t[1]+t[2]+t[3])

def manip_norm(s):
    return NUM.manipulabilite(s)/(LC**3)

# ═══════════════════════════════════════════════════════════════
# DESSIN 3D
# ═══════════════════════════════════════════════════════════════
def cylinder(p0,p1,r,n=12):
    v=p1-p0; L=np.linalg.norm(v)
    if L<1e-6: return []
    v/=L
    arb=np.array([1,0,0]) if abs(v[0])<0.9 else np.array([0,1,0])
    u=np.cross(v,arb); u/=np.linalg.norm(u); w=np.cross(v,u)
    a=np.linspace(0,2*np.pi,n,endpoint=False)
    circ=[r*(np.cos(x)*u+np.sin(x)*w) for x in a]
    return [[p0+circ[i],p0+circ[(i+1)%n],p1+circ[(i+1)%n],p1+circ[i]] for i in range(n)]

def draw_arm(ax,pts,color,alpha,objs):
    for i in range(len(pts)-1):
        f=cylinder(pts[i],pts[i+1],4.5)
        if f:
            col=Poly3DCollection(f,alpha=alpha*0.9,linewidth=0)
            col.set_facecolor(color); ax.add_collection3d(col); objs.append(col)
    for i,p in enumerate(pts):
        if i==0:
            sc=ax.scatter(*p,s=120,color=C_JNT,alpha=alpha,depthshade=False,zorder=5)
        elif i==len(pts)-1:
            sc=ax.scatter(*p,s=160,color=color,alpha=alpha,depthshade=False,zorder=6,
                          edgecolors='white',linewidths=0.5)
        else:
            sc=ax.scatter(*p,s=60,color=C_JNT,alpha=alpha,depthshade=False,zorder=5,
                          edgecolors=color,linewidths=0.8)
        objs.append(sc)
    ef=pts[-1]
    objs.append(ax.text(ef[0]+10,ef[1]+10,ef[2]+10,
        '({:.0f},{:.0f},{:.0f})'.format(*ef),
        color=color,fontsize=7.5,alpha=alpha,fontfamily='monospace',fontweight='bold'))

def init_ax(ax,title,color):
    ax.set_facecolor(BG)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor('#0D1520')
    ax.tick_params(colors='#0D1520',labelsize=5)
    ax.set_xlabel('X',fontsize=6,color='#C0392B',labelpad=1)
    ax.set_ylabel('Y',fontsize=6,color=C_ALG,labelpad=1)
    ax.set_zlabel('Z',fontsize=6,color=C_NUM,labelpad=1)
    R=450
    ax.set_xlim(-R,R);ax.set_ylim(-R,R);ax.set_zlim(0,R)
    ax.set_title(title,color=color,fontsize=8.5,pad=4,fontfamily='monospace',fontweight='bold')
    ax.view_init(elev=24,azim=-52)
    for g in np.arange(-400,401,100):
        ax.plot([g,g],[-400,400],[0,0],color='#0D1520',lw=0.35,zorder=0)
        ax.plot([-400,400],[g,g],[0,0],color='#0D1520',lw=0.35,zorder=0)
    theta=np.linspace(0,2*np.pi,80)
    R_max=L2+L3+L4+L5+L6
    ax.plot(R_max*np.cos(theta),R_max*np.sin(theta),np.zeros(80),
            color=C_JNT,lw=0.4,alpha=0.3,zorder=0)

# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
plt.rcParams['font.family']='monospace'
fig=plt.figure(figsize=(23,11),facecolor=BG)
fig.canvas.manager.set_window_title('xarm-ik-explorer  v1.0  —  ALG vs NUM')

gs=gridspec.GridSpec(1,2,left=0.003,right=0.593,top=0.97,bottom=0.03,wspace=0.06)
ax_a=fig.add_subplot(gs[0],projection='3d')
ax_n=fig.add_subplot(gs[1],projection='3d')
init_ax(ax_a,'[ ALG ]  Analytique  ·  Al-Kashi',C_ALG)
init_ax(ax_n,'[ NUM ]  Numérique   ·  DLS / Jacobien',C_NUM)
oa,on_=[],[]

fig.add_axes([0.597,0.01,0.002,0.97],facecolor='#1A2332').set_axis_off()

P=0.609; W=0.384

# ── En-tête panel ──
# Titre centré sur toute la largeur de l'écran
fig.text(0.5, 0.976, 'xarm-ik-explorer',
    color='#2A5A8A', fontsize=15, fontweight='bold',
    fontfamily='monospace', ha='center', va='center')
# CONTROL PANEL centré sur la colonne droite
fig.text(P+W/2, 0.958, 'CONTROL  PANEL',
    color='#1A2A3A', fontsize=7.5,
    fontfamily='monospace', ha='center', va='center')
fig.add_axes([P,0.950,W,0.0015],facecolor=C_ALG).set_axis_off()

# ═══════════════════════════════════════════════════════════════
# MODE TOGGLE  — deux boutons, look onglets
# ═══════════════════════════════════════════════════════════════
ax_bfk=fig.add_axes([P,       0.912,W/2-0.004,0.030])
ax_bik=fig.add_axes([P+W/2+0.004,0.912,W/2-0.004,0.030])
btn_fk=widgets.Button(ax_bfk,'⬤  FK  Cinématique Directe',color='#071507',hovercolor='#0D2A0D')
btn_ik=widgets.Button(ax_bik,'⬤  IK  Cinématique Inverse',color='#07070F',hovercolor='#0D1530')
btn_fk.label.set_color(C_ALG); btn_fk.label.set_fontsize(8); btn_fk.label.set_fontweight('bold')
btn_ik.label.set_color('#2A5A7A'); btn_ik.label.set_fontsize(8)
fig.add_axes([P,0.906,W,0.0012],facecolor=C_SEP).set_axis_off()

# ═══════════════════════════════════════════════════════════════
# ZONE DE CONTRÔLE UNIQUE  [P, Y_CTRL_BOT, W, Y_CTRL_H]
# Les deux sections (FK et IK) occupent exactement la même zone.
# On affiche l'une OU l'autre selon le mode.
# ═══════════════════════════════════════════════════════════════
Y_CTRL_TOP = 0.900
Y_DASH_TOP = 0.450   # le dashboard commence ici
Y_CTRL_H   = Y_CTRL_TOP - Y_DASH_TOP

# ── Fond zone contrôle ──
ax_ctrl_bg=fig.add_axes([P,Y_DASH_TOP,W,Y_CTRL_H],facecolor='#08090F')
ax_ctrl_bg.set_axis_off()

# ════════════ SECTION FK ════════════
# Header FK
ax_fk_hdr=fig.add_axes([P,0.878,W,0.022],facecolor='#061506')
ax_fk_hdr.set_axis_off()
t_fk_hdr=ax_fk_hdr.text(0.03,0.5,'FK  —  CONTRÔLE JOINTS',
    transform=ax_fk_hdr.transAxes,
    color=C_ALG,fontsize=7.5,fontweight='bold',fontfamily='monospace',va='center')

# Sliders FK — centrés dans la zone
sliders_fk=[]
SL_H=0.024; SL_ST=0.052
Y_SL_START=0.868
for i in range(5):
    y=Y_SL_START - i*SL_ST
    ax_s=fig.add_axes([P,y,W,SL_H],facecolor='#090D09')
    sl=widgets.Slider(ax_s,NOMS[i],LIMITS[i][0],LIMITS[i][1],
                      valinit=HOME[i],valstep=1,color=C_ALG)
    sl.label.set_color(C_ALG); sl.label.set_fontsize(7.5)
    sl.valtext.set_color('#4A8A6A'); sl.valtext.set_fontsize(9)
    sl.poly.set_alpha(0.6)
    sliders_fk.append(sl)

# Label hint FK
ax_fk_hint=fig.add_axes([P,Y_SL_START-5*SL_ST-0.005,W,0.020],facecolor='#08090F')
ax_fk_hint.set_axis_off()
ax_fk_hint.text(0.5,0.5,
    'Déplacez les curseurs pour explorer la cinématique directe',
    transform=ax_fk_hint.transAxes,
    color='#1A3A2A',fontsize=6.5,fontfamily='monospace',va='center',ha='center')

# ════════════ SECTION IK ════════════
# Header IK
ax_ik_hdr=fig.add_axes([P,0.878,W,0.022],facecolor='#06060F')
ax_ik_hdr.set_axis_off()
t_ik_hdr=ax_ik_hdr.text(0.03,0.5,'IK  —  SAISIE CIBLE',
    transform=ax_ik_hdr.transAxes,
    color=C_NUM,fontsize=7.5,fontweight='bold',fontfamily='monospace',va='center')

# XYZ — 3 champs larges
TB_W=(W-0.012)/3.0; TB_H=0.042
Yxyz=0.800

# Labels XYZ (Text objets — stockés pour les masquer)
lbl_x=fig.text(P+0.000,          Yxyz+0.048,'X  (mm)',color='#2A3A5A',fontsize=7,fontfamily='monospace')
lbl_y=fig.text(P+TB_W+0.006,     Yxyz+0.048,'Y  (mm)',color='#2A3A5A',fontsize=7,fontfamily='monospace')
lbl_z=fig.text(P+2*(TB_W+0.006), Yxyz+0.048,'Z  (mm)',color='#2A3A5A',fontsize=7,fontfamily='monospace')

ax_tx=fig.add_axes([P,               Yxyz,TB_W,TB_H],facecolor='#090B12')
ax_ty=fig.add_axes([P+TB_W+0.006,    Yxyz,TB_W,TB_H],facecolor='#090B12')
ax_tz=fig.add_axes([P+2*(TB_W+0.006),Yxyz,TB_W,TB_H],facecolor='#090B12')
tb_x=widgets.TextBox(ax_tx,'','0',  color='#090B12',hovercolor='#121820')
tb_y=widgets.TextBox(ax_ty,'','164',color='#090B12',hovercolor='#121820')
tb_z=widgets.TextBox(ax_tz,'','273',color='#090B12',hovercolor='#121820')
for tb in [tb_x,tb_y,tb_z]:
    tb.text_disp.set_color('#C8D8E8'); tb.text_disp.set_fontsize(12)

# PHI indicateurs (lecture seule)
PH_W=(W-0.008)/2.0; PH_H=0.022
Yphi=0.724
lbl_phia=fig.text(P,           Yphi+0.028,'PHI  ALG  °  (lecture seule)',
    color='#1A4A3A',fontsize=6,fontfamily='monospace')
lbl_phin=fig.text(P+PH_W+0.008,Yphi+0.028,'PHI  NUM  °  (lecture seule)',
    color='#1A3A5A',fontsize=6,fontfamily='monospace')

ax_pa=fig.add_axes([P,           Yphi,PH_W,PH_H],facecolor='#090D09')
ax_pn=fig.add_axes([P+PH_W+0.008,Yphi,PH_W,PH_H],facecolor='#090B12')
sl_phi_a=widgets.Slider(ax_pa,'',-180,180,valinit=0,valstep=0.1,color=C_ALG)
sl_phi_n=widgets.Slider(ax_pn,'',-180,180,valinit=0,valstep=0.1,color=C_NUM)
for sl in [sl_phi_a,sl_phi_n]:
    sl.label.set_color('#111')
    sl.valtext.set_color('#3A5A4A'); sl.valtext.set_fontsize(8)
    sl.poly.set_alpha(0.25); sl.set_active(False)

# GO IK / RESET
BTN_H=0.042; BTN_W=(W-0.008)/2.0
Ybtn=0.644
ax_go =fig.add_axes([P,            Ybtn,BTN_W,BTN_H])
ax_rst=fig.add_axes([P+BTN_W+0.008,Ybtn,BTN_W,BTN_H])
btn_go =widgets.Button(ax_go, 'GO  IK',color='#071A07',hovercolor='#0D2A0D')
btn_rst=widgets.Button(ax_rst,'RESET', color='#0D0D0D',hovercolor='#1A1A1A')
btn_go.label.set_color(C_ALG); btn_go.label.set_fontsize(12); btn_go.label.set_fontweight('bold')
btn_rst.label.set_color('#3A4A5A'); btn_rst.label.set_fontsize(9)

# Hint IK
ax_ik_hint=fig.add_axes([P,0.606,W,0.030],facecolor='#08090F')
ax_ik_hint.set_axis_off()
ax_ik_hint.text(0.5,0.5,
    'Entrez X Y Z en mm  ·  GO IK lance les deux solveurs en parallèle',
    transform=ax_ik_hint.transAxes,
    color='#1A2A4A',fontsize=6.5,fontfamily='monospace',va='center',ha='center')

# ═══════════════════════════════════════════════════════════════
# SÉPARATEUR avant dashboard
# ═══════════════════════════════════════════════════════════════
fig.add_axes([P,Y_DASH_TOP-0.001,W,0.0012],facecolor=C_SEP).set_axis_off()

# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════
# ── Dimensions dashboard / log ──
# Dashboard : colonne gauche  P → P + DASH_W
# Log       : colonne droite  P + DASH_W + GAP_COL → P + W
DASH_W   = W * 0.56      # largeur dashboard
GAP_COL  = 0.008         # gap entre dashboard et log
LOG_W    = W - DASH_W - GAP_COL  # largeur log
LOG_X    = P + DASH_W + GAP_COL  # X origine log
Y_BOT    = 0.010         # bas commun
Y_HDR_H  = 0.020         # hauteur header
BLOC_H   = Y_DASH_TOP - 0.020 - Y_BOT  # hauteur totale du bloc

# ── Header commun (pleine largeur) ──
ax_dash_hdr=fig.add_axes([P,Y_DASH_TOP-Y_HDR_H,W,Y_HDR_H],facecolor='#0A0D12')
ax_dash_hdr.set_axis_off()
ax_dash_hdr.text(0.03,0.5,'DASHBOARD  —  ÉTAT EN TEMPS RÉEL',
    transform=ax_dash_hdr.transAxes,
    color='#2A3A4A',fontsize=7,fontweight='bold',fontfamily='monospace',va='center')
ax_dash_hdr.text(0.70,0.5,'DEBUG  LOG',
    transform=ax_dash_hdr.transAxes,
    color='#1A3A1A',fontsize=7,fontweight='bold',fontfamily='monospace',va='center')

# ── Dashboard (colonne gauche) ──
ax_dash=fig.add_axes([P, Y_BOT, DASH_W, BLOC_H],facecolor='#060A0F')
ax_dash.set_axis_off()
dash_txt=ax_dash.text(0.025,0.99,'',transform=ax_dash.transAxes,
    color=C_TXT,fontsize=7.0,va='top',fontfamily='monospace',linespacing=1.35)

# ── Séparateur vertical dashboard / log ──
fig.add_axes([P+DASH_W+0.002, Y_BOT, 0.0012, BLOC_H],facecolor=C_SEP).set_axis_off()

# ── Console Debug (colonne droite, même hauteur) ──
ax_log=fig.add_axes([LOG_X, Y_BOT, LOG_W, BLOC_H],facecolor=C_LOG_BG)
ax_log.set_axis_off()
log_txt=ax_log.text(0.06,0.97,'',transform=ax_log.transAxes,
    color=C_LOG_TX,fontsize=7.5,va='top',fontfamily='monospace',linespacing=1.40)

sys.stdout=StdoutCapture()

# ═══════════════════════════════════════════════════════════════
# DASHBOARD RENDER
# ═══════════════════════════════════════════════════════════════
SEP='-'*38

def fv(v,d=2):
    return ('{:.'+str(d)+'f}').format(v) if v is not None else '      --'

def status_str(s):
    if s=='OK':    return ' ✓ OK  '
    if s=='ECHEC': return ' ✗ FAIL'
    if s=='FK':    return ' ~ FK  '
    return '  '+str(s)

def render_dash(sa,sn):
    pa=get_pts_alg(sa)[-1]; pn=get_pts_num(sn)[-1]
    mva=manip_norm(sa); mvn=manip_norm(sn)
    pha=phi_reel(sa); phn=phi_reel(sn)
    tgt=target_pt[0]
    tline=('Cible   {:>8.1f}  {:>8.1f}  {:>8.1f} mm'.format(*tgt)
           if tgt else 'Cible        --       --       --')
    c=C_TXT if dash['st_a']!='ECHEC' or dash['st_n']!='ECHEC' else '#4A2A2A'
    dash_txt.set_color(c)
    dash_txt.set_text((
        '          ANALYTIQUE    NUMERIQUE\n'
        '{sep}\n'
        'Statut   {sa:>10s}  {sn:>10s}\n'
        'Err  mm  {ea:>10s}  {en:>10s}\n'
        'T    ms  {ta:>10s}  {tn:>10s}\n'
        '{sep}\n'
        '{tl}\n'
        '{sep}\n'
        '         EFF  ALG     EFF  NUM\n'
        'X  mm    {ax:>10.2f}  {nx:>10.2f}\n'
        'Y  mm    {ay:>10.2f}  {ny:>10.2f}\n'
        'Z  mm    {az:>10.2f}  {nz:>10.2f}\n'
        '{sep}\n'
        'Manip    {ma:>10.4f}  {mn:>10.4f}\n'
        'Phi  °   {pa:>10.1f}  {pn:>10.1f}\n'
        '{sep}\n'
        '            ALG           NUM\n'
        'J1  °    {j1a:>10.1f}  {j1n:>10.1f}\n'
        'J2  °    {j2a:>10.1f}  {j2n:>10.1f}\n'
        'J3  °    {j3a:>10.1f}  {j3n:>10.1f}\n'
        'J4  °    {j4a:>10.1f}  {j4n:>10.1f}\n'
        'J5  °    {j5a:>10.1f}  {j5n:>10.1f}\n'
    ).format(sep=SEP,
        sa=status_str(dash['st_a']),sn=status_str(dash['st_n']),
        ea=fv(dash['err_a']),en=fv(dash['err_n']),
        ta=fv(dash['t_a'],1),tn=fv(dash['t_n'],1),
        tl=tline,
        ax=pa[0],nx=pn[0],ay=pa[1],ny=pn[1],az=pa[2],nz=pn[2],
        ma=mva,mn=mvn,pa=pha,pn=phn,
        j1a=sa[0],j1n=sn[0],j2a=sa[1],j2n=sn[1],
        j3a=sa[2],j3n=sn[2],j4a=sa[3],j4n=sn[3],
        j5a=sa[4],j5n=sn[4]))

# ═══════════════════════════════════════════════════════════════
# REDRAW
# ═══════════════════════════════════════════════════════════════
def redraw(sa=None,sn=None):
    global oa,on_
    for o in oa:
        try: o.remove()
        except: pass
    for o in on_:
        try: o.remove()
        except: pass
    oa,on_=[],[]
    sa=np.array(sa if sa is not None else angles_alg,dtype=float)
    sn=np.array(sn if sn is not None else angles_num,dtype=float)
    draw_arm(ax_a,get_pts_alg(sa),C_ALG,1.0,oa)
    draw_arm(ax_n,get_pts_num(sn),C_NUM,1.0,on_)
    if target_pt[0]:
        for ax,ol in [(ax_a,oa),(ax_n,on_)]:
            ol.append(ax.scatter(*target_pt[0],s=260,color=C_TGT,
                                  marker='*',depthshade=False,zorder=10,
                                  edgecolors='white',linewidths=0.4))
    render_dash(sa,sn)
    fig.canvas.draw_idle()

# ═══════════════════════════════════════════════════════════════
# GESTION MODE  — show/hide via set_visible sur TOUS les objets
# ═══════════════════════════════════════════════════════════════
# Tous les objets de chaque section
_fk_objects = ([sl.ax for sl in sliders_fk] +
               [ax_fk_hdr, ax_fk_hint] +
               [sl.label for sl in sliders_fk] +
               [sl.valtext for sl in sliders_fk])

_ik_objects = ([ax_tx, ax_ty, ax_tz, ax_pa, ax_pn,
                ax_go, ax_rst, ax_ik_hdr, ax_ik_hint] +
               [lbl_x, lbl_y, lbl_z, lbl_phia, lbl_phin])

def set_mode(m):
    mode[0]=m
    is_fk=(m=='FK')

    # Show/hide FK
    for obj in _fk_objects:
        obj.set_visible(is_fk)

    # Show/hide IK
    for obj in _ik_objects:
        obj.set_visible(not is_fk)

    # Activer/désactiver widgets IK
    for tb in [tb_x,tb_y,tb_z]:
        try: tb.set_active(not is_fk)
        except: pass
    try: btn_go.set_active(not is_fk)
    except: pass

    # Style boutons toggle
    if is_fk:
        btn_fk.label.set_color(C_ALG);   btn_fk.label.set_fontweight('bold')
        btn_ik.label.set_color('#2A5A7A'); btn_ik.label.set_fontweight('normal')
        btn_fk.color='#071507'; btn_ik.color='#07070F'
    else:
        btn_ik.label.set_color(C_NUM);   btn_ik.label.set_fontweight('bold')
        btn_fk.label.set_color('#2A5A7A'); btn_fk.label.set_fontweight('normal')
        btn_ik.color='#07070F'; btn_fk.color='#070707'

    # Reset flag hors workspace quand on change de mode
    _ik_failed[0]=False

    fig.canvas.draw_idle()

# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════
def on_mode_fk(e): set_mode('FK')
def on_mode_ik(e): set_mode('IK')

def on_slider_fk(val):
    if mode[0]!='FK': return
    for i,sl in enumerate(sliders_fk):
        angles_fk[i]=sl.val
    angles_alg[:]=list(angles_fk)
    angles_num[:]=list(angles_fk)
    _guard[0]=True
    sl_phi_a.set_val(round(phi_reel(angles_fk),1))
    sl_phi_n.set_val(round(phi_reel(angles_fk),1))
    _guard[0]=False
    target_pt[0]=None
    _ik_failed[0]=False
    dash.update(st_a='FK',st_n='FK',err_a=None,err_n=None,t_a=None,t_n=None)
    redraw()

def go_ik(e):
    if mode[0]!='IK': return

    # Reset flag hors workspace avant chaque GO
    _ik_failed[0]=False

    try:
        X=float(tb_x.text); Y=float(tb_y.text); Z=float(tb_z.text)
    except:
        log('[ERR]  Coordonnées invalides'); return

    target_pt[0]=[X,Y,Z]
    log('[IK]   Cible  X={:.1f}  Y={:.1f}  Z={:.1f}'.format(X,Y,Z))

    # ALG
    t0=time.perf_counter()
    res_a,phi_u=ALG.ik_auto_full(X,Y,Z,phi_deg=0,theta5_servo_deg=angles_alg[4])
    t_alg=(time.perf_counter()-t0)*1000
    if res_a is not None:
        log('[ALG]  OK  phi={}°  t={:.1f}ms'.format(phi_u,t_alg))
    else:
        log('[ALG]  ECHEC  t={:.1f}ms'.format(t_alg))

    # NUM
    t0=time.perf_counter()
    res_n=NUM.ik(X,Y,Z,theta5_servo_deg=angles_num[4],
                 angles_depart=list(angles_num),verbose=True)
    t_num=(time.perf_counter()-t0)*1000
    if res_n is not None:
        log('[NUM]  OK  t={:.1f}ms'.format(t_num))
    else:
        log('[NUM]  ECHEC  t={:.1f}ms'.format(t_num))

    # Si les DEUX échouent → rien ne bouge, on affiche juste ECHEC
    if res_a is None and res_n is None:
        _ik_failed[0]=True
        log('[SYS]  Position hors workspace — bras immobile')
        dash.update(st_a='ECHEC',st_n='ECHEC',err_a=None,err_n=None,
                    t_a=t_alg,t_n=t_num)
        # On redraw SANS changer les angles — bras reste où il est
        redraw(angles_alg,angles_num)
        return

    sa1=np.array(res_a if res_a is not None else angles_alg,dtype=float)
    sn1=np.array(res_n if res_n is not None else angles_num, dtype=float)
    sa0=np.array(angles_alg,dtype=float)
    sn0=np.array(angles_num, dtype=float)

    # Animation
    global oa,on_
    for step in range(N_ANIM+1):
        alpha=step/N_ANIM
        sa_=sa0+alpha*(sa1-sa0); sn_=sn0+alpha*(sn1-sn0)
        for o in oa:
            try: o.remove()
            except: pass
        for o in on_:
            try: o.remove()
            except: pass
        oa,on_=[],[]
        draw_arm(ax_a,get_pts_alg(sa_),C_ALG,1.0,oa)
        draw_arm(ax_n,get_pts_num(sn_),C_NUM,1.0,on_)
        if target_pt[0]:
            for ax,ol in [(ax_a,oa),(ax_n,on_)]:
                ol.append(ax.scatter(*target_pt[0],s=260,color=C_TGT,
                                      marker='*',depthshade=False,zorder=10))
        fig.canvas.flush_events(); plt.pause(0.014)

    # Mise à jour état SEULEMENT si succès
    if res_a is not None: angles_alg[:]=list(sa1)
    if res_n is not None: angles_num[:]=list(sn1)

    _guard[0]=True
    sl_phi_a.set_val(round(phi_reel(angles_alg),1))
    sl_phi_n.set_val(round(phi_reel(angles_num),1))
    _guard[0]=False

    xyz_a,_   =ALG.fk(sa1)
    xyz_n,_,_ =NUM.fk(sn1)
    tgt=np.array([X,Y,Z])
    err_a=np.linalg.norm(tgt-xyz_a) if res_a is not None else None
    err_n=np.linalg.norm(tgt-xyz_n) if res_n is not None else None

    if err_a is not None: log('[ALG]  FK err = {:.3f} mm'.format(err_a))
    if err_n is not None: log('[NUM]  FK err = {:.3f} mm'.format(err_n))

    dash.update(
        st_a='OK'    if res_a is not None else 'ECHEC',
        st_n='OK'    if res_n is not None else 'ECHEC',
        err_a=err_a,err_n=err_n,t_a=t_alg,t_n=t_num)
    redraw(sa1,sn1)

def reset(e):
    target_pt[0]=None
    _ik_failed[0]=False
    angles_fk[:]=list(HOME)
    angles_alg[:]=list(HOME)
    angles_num[:]=list(HOME)
    dash.update(st_a='--',st_n='--',err_a=None,err_n=None,t_a=None,t_n=None)
    _guard[0]=True
    for i,sl in enumerate(sliders_fk): sl.set_val(HOME[i])
    sl_phi_a.set_val(0.0); sl_phi_n.set_val(0.0)
    _guard[0]=False
    log('[SYS]  Reset  →  HOME [90,90,90,0,90]')
    redraw()

# ═══════════════════════════════════════════════════════════════
# BRANCHEMENT & INIT
# ═══════════════════════════════════════════════════════════════
btn_fk.on_clicked(on_mode_fk)
btn_ik.on_clicked(on_mode_ik)
for sl in sliders_fk: sl.on_changed(on_slider_fk)
btn_go.on_clicked(go_ik)
btn_rst.on_clicked(reset)

log('[SYS]  xarm-ik-explorer  v1.0  démarré')
log('[SYS]  Mode  :  FK  —  Cinématique Directe')
log('[SYS]  HOME  :  [90,90,90,0,90]  →  [0, 164, 273] mm')
set_mode('FK')
redraw()
plt.show()
