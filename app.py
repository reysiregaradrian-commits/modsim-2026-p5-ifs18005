import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #DBEAFE;
        padding-bottom: 4px;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stage-card {
        background-color: #F8FAFC;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 4px solid #10B981;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================
class ProjectStage:
    """Memodelkan tahapan konstruksi gedung dengan faktor risiko"""

    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []

    def sample_duration(self, n_simulations):
        base_duration = np.random.triangular(
            self.optimistic,
            self.most_likely,
            self.pessimistic,
            n_simulations
        )
        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                probability = risk_params['probability']
                impact = risk_params['impact']
                risk_occurs = np.random.random(n_simulations) < probability
                base_duration = np.where(
                    risk_occurs,
                    base_duration * (1 + impact),
                    base_duration
                )
            elif risk_params['type'] == 'continuous':
                mean = risk_params['mean']
                std = risk_params['std']
                productivity_factor = np.random.normal(mean, std, n_simulations)
                base_duration = base_duration / np.clip(productivity_factor, 0.5, 1.5)
        return base_duration


class MonteCarloConstructionSim:
    """Simulasi Monte Carlo untuk proyek konstruksi gedung"""

    def __init__(self, stages_config, num_simulations=10000):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.stages = {}
        self.simulation_results = None
        self._init_stages()

    def _init_stages(self):
        for stage_name, config in self.stages_config.items():
            self.stages[stage_name] = ProjectStage(
                name=stage_name,
                base_params=config['base_params'],
                risk_factors=config.get('risk_factors', {}),
                dependencies=config.get('dependencies', [])
            )

    def run_simulation(self):
        np.random.seed(42)
        results = pd.DataFrame(index=range(self.num_simulations))

        for stage_name, stage in self.stages.items():
            results[stage_name] = stage.sample_duration(self.num_simulations)

        start_times = pd.DataFrame(index=range(self.num_simulations))
        end_times = pd.DataFrame(index=range(self.num_simulations))

        for stage_name in self.stages.keys():
            deps = self.stages[stage_name].dependencies
            if not deps:
                start_times[stage_name] = 0
            else:
                start_times[stage_name] = end_times[deps].max(axis=1)
            end_times[stage_name] = start_times[stage_name] + results[stage_name]

        results['Total_Duration'] = end_times.max(axis=1)

        for stage_name in self.stages.keys():
            results[f'{stage_name}_Start'] = start_times[stage_name]
            results[f'{stage_name}_Finish'] = end_times[stage_name]

        self.simulation_results = results
        return results

    def critical_path_prob(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        total_duration = self.simulation_results['Total_Duration']
        result = {}
        for stage_name in self.stages.keys():
            stage_finish = self.simulation_results[f'{stage_name}_Finish']
            correlation = self.simulation_results[stage_name].corr(total_duration)
            is_critical = (stage_finish + 0.01) >= total_duration
            result[stage_name] = {
                'probability': float(np.mean(is_critical)),
                'correlation': float(correlation),
                'avg_duration': float(self.simulation_results[stage_name].mean())
            }
        return pd.DataFrame(result).T

    def risk_contribution(self):
        if self.simulation_results is None:
            raise ValueError("Run simulation first")
        total_var = self.simulation_results['Total_Duration'].var()
        contrib = {}
        for stage_name in self.stages.keys():
            stage_covar = self.simulation_results[stage_name].cov(
                self.simulation_results['Total_Duration']
            )
            contrib[stage_name] = {
                'variance': float(self.simulation_results[stage_name].var()),
                'contribution_percent': float((stage_covar / total_var) * 100),
                'std_dev': float(self.simulation_results[stage_name].std())
            }
        return pd.DataFrame(contrib).T


# ============================================================================
# 3. KONFIGURASI TAHAPAN KONSTRUKSI GEDUNG FITE
# ============================================================================
DEFAULT_STAGES_CONFIG = {
    "Persiapan_Lahan": {
        "base_params": {"optimistic": 1, "most_likely": 2, "pessimistic": 3},
        "risk_factors": {
            "cuaca_buruk": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.3
            },
            "perizinan": {
                "type": "discrete",
                "probability": 0.2,
                "impact": 0.5
            }
        }
    },
    "Pondasi": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "kondisi_tanah": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.2
            },
            "cuaca_buruk": {
                "type": "discrete",
                "probability": 0.3,
                "impact": 0.25
            }
        },
        "dependencies": ["Persiapan_Lahan"]
    },
    "Struktur_Beton": {
        "base_params": {"optimistic": 5, "most_likely": 7, "pessimistic": 10},
        "risk_factors": {
            "keterlambatan_material": {
                "type": "discrete",
                "probability": 0.35,
                "impact": 0.2
            },
            "produktivitas_pekerja": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.25
            },
            "cuaca_buruk": {
                "type": "discrete",
                "probability": 0.3,
                "impact": 0.15
            }
        },
        "dependencies": ["Pondasi"]
    },
    "Dinding_dan_Atap": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "keterlambatan_material": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.2
            },
            "produktivitas_pekerja": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.2
            }
        },
        "dependencies": ["Struktur_Beton"]
    },
    "Instalasi_MEP": {
        "base_params": {"optimistic": 3, "most_likely": 4, "pessimistic": 7},
        "risk_factors": {
            "keterlambatan_material_teknis": {
                "type": "discrete",
                "probability": 0.4,
                "impact": 0.3
            },
            "kompleksitas_instalasi": {
                "type": "continuous",
                "mean": 1.0,
                "std": 0.3
            }
        },
        "dependencies": ["Struktur_Beton"]
    },
    "Interior_dan_Finishing": {
        "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
        "risk_factors": {
            "perubahan_desain_lab": {
                "type": "discrete",
                "probability": 0.35,
                "impact": 0.4
            },
            "keterlambatan_material": {
                "type": "discrete",
                "probability": 0.25,
                "impact": 0.2
            }
        },
        "dependencies": ["Dinding_dan_Atap", "Instalasi_MEP"]
    },
    "Komisioning_dan_Serahterima": {
        "base_params": {"optimistic": 1, "most_likely": 1, "pessimistic": 2},
        "risk_factors": {
            "temuan_inspeksi": {
                "type": "discrete",
                "probability": 0.2,
                "impact": 0.8
            }
        },
        "dependencies": ["Interior_dan_Finishing"]
    }
}

STAGE_LABELS = {
    "Persiapan_Lahan": "Persiapan Lahan",
    "Pondasi": "Pondasi",
    "Struktur_Beton": "Struktur Beton (5 lantai)",
    "Dinding_dan_Atap": "Dinding & Atap",
    "Instalasi_MEP": "Instalasi MEP",
    "Interior_dan_Finishing": "Interior & Finishing Lab",
    "Komisioning_dan_Serahterima": "Komisioning & Serahterima"
}

DEADLINE_SCENARIOS = [16, 20, 24]  # dalam bulan


# ============================================================================
# 4. FUNGSI VISUALISASI PLOTLY
# ============================================================================
def plot_distribution(results):
    total = results['Total_Duration']
    mean_d = total.mean()
    median_d = float(np.median(total))
    ci80 = np.percentile(total, [10, 90])
    ci95 = np.percentile(total, [2.5, 97.5])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=total,
        nbinsx=60,
        name='Distribusi Durasi',
        marker_color='#3B82F6',
        opacity=0.7,
        histnorm='probability density'
    ))
    fig.add_vline(x=mean_d, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_d:.1f} bln", annotation_position="top right")
    fig.add_vline(x=median_d, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_d:.1f} bln")
    fig.add_vrect(x0=ci80[0], x1=ci80[1], fillcolor="#FDE68A", opacity=0.25,
                  annotation_text="80% CI", line_width=0)
    fig.add_vrect(x0=ci95[0], x1=ci95[1], fillcolor="#FCA5A5", opacity=0.1,
                  annotation_text="95% CI", line_width=0)
    # Deadline scenarios
    colors_dl = ['#10B981', '#F59E0B', '#EF4444']
    for dl, col in zip(DEADLINE_SCENARIOS, colors_dl):
        fig.add_vline(x=dl, line_dash="dot", line_color=col,
                      annotation_text=f"{dl} bln", annotation_position="top left")

    fig.update_layout(
        title='Distribusi Total Durasi Pembangunan Gedung FITE',
        xaxis_title='Durasi Total (Bulan)',
        yaxis_title='Densitas Probabilitas',
        height=480,
        legend=dict(orientation="h")
    )
    return fig, {'mean': mean_d, 'median': median_d, 'std': float(total.std()),
                 'min': float(total.min()), 'max': float(total.max()),
                 'ci80': ci80, 'ci95': ci95}


def plot_completion_prob(results):
    deadlines = np.arange(10, 32, 0.5)
    probs = [float(np.mean(results['Total_Duration'] <= d)) for d in deadlines]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deadlines, y=probs,
        mode='lines',
        name='Probabilitas Selesai',
        line=dict(color='#1E40AF', width=3),
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.15)'
    ))
    for h, col, label in [(0.5, '#EF4444', '50%'), (0.8, '#10B981', '80%'), (0.95, '#6366F1', '95%')]:
        fig.add_hline(y=h, line_dash="dash", line_color=col,
                      annotation_text=label, annotation_position="right")

    colors_dl = ['#10B981', '#F59E0B', '#EF4444']
    labels_dl = ['16 bln', '20 bln', '24 bln']
    for dl, col, lbl in zip(DEADLINE_SCENARIOS, colors_dl, labels_dl):
        idx = np.argmin(np.abs(deadlines - dl))
        prob = probs[idx]
        fig.add_trace(go.Scatter(
            x=[dl], y=[prob],
            mode='markers+text',
            marker=dict(size=14, color=col, line=dict(color='white', width=2)),
            text=[f'{prob:.1%}'],
            textposition='top center',
            name=lbl
        ))

    fig.update_layout(
        title='Kurva Probabilitas Penyelesaian Proyek',
        xaxis_title='Deadline (Bulan)',
        yaxis_title='Probabilitas Selesai Tepat Waktu',
        yaxis_range=[-0.05, 1.05],
        xaxis_range=[10, 32],
        height=480
    )
    return fig


def plot_critical_path(critical_df):
    df = critical_df.sort_values('probability', ascending=True)
    colors = ['#EF4444' if p > 0.7 else '#FCA5A5' if p > 0.4 else '#93C5FD'
              for p in df['probability']]
    fig = go.Figure(go.Bar(
        y=[STAGE_LABELS.get(s, s) for s in df.index],
        x=df['probability'],
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1%}' for p in df['probability']],
        textposition='auto'
    ))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.7, line_dash="dot", line_color="orange",
                  annotation_text="Kritis (>70%)")
    fig.update_layout(
        title='Probabilitas Critical Path per Tahapan',
        xaxis_title='Probabilitas Menjadi Critical Path',
        xaxis_range=[0, 1.0],
        height=420
    )
    return fig


def plot_boxplot(results, stages):
    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    for i, stage_name in enumerate(stages.keys()):
        fig.add_trace(go.Box(
            y=results[stage_name],
            name=STAGE_LABELS.get(stage_name, stage_name),
            boxmean='sd',
            marker_color=palette[i % len(palette)],
            boxpoints='outliers'
        ))
    fig.update_layout(
        title='Distribusi Durasi per Tahapan Konstruksi',
        yaxis_title='Durasi (Bulan)',
        height=440,
        showlegend=False
    )
    return fig


def plot_risk_contribution(risk_df):
    df = risk_df.sort_values('contribution_percent', ascending=False)
    fig = go.Figure(go.Bar(
        x=[STAGE_LABELS.get(s, s) for s in df.index],
        y=df['contribution_percent'],
        marker_color=px.colors.qualitative.Pastel,
        text=[f'{v:.1f}%' for v in df['contribution_percent']],
        textposition='auto'
    ))
    fig.update_layout(
        title='Kontribusi Risiko per Tahapan terhadap Variabilitas Total',
        yaxis_title='Kontribusi (%)',
        height=400
    )
    return fig


def plot_gantt_mean(results, stages):
    """Gantt chart berdasarkan rata-rata durasi"""
    rows = []
    for stage_name in stages.keys():
        start_col = f'{stage_name}_Start'
        finish_col = f'{stage_name}_Finish'
        rows.append({
            'Tahapan': STAGE_LABELS.get(stage_name, stage_name),
            'Mulai': float(results[start_col].mean()),
            'Selesai': float(results[finish_col].mean()),
        })
    df = pd.DataFrame(rows)
    fig = go.Figure()
    palette = px.colors.qualitative.Set3
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Selesai'] - row['Mulai']],
            y=[row['Tahapan']],
            base=[row['Mulai']],
            orientation='h',
            marker_color=palette[i % len(palette)],
            name=row['Tahapan'],
            text=[f"{row['Selesai'] - row['Mulai']:.1f} bln"],
            textposition='inside',
            insidetextanchor='middle'
        ))
    # Deadline lines
    colors_dl = ['#10B981', '#F59E0B', '#EF4444']
    for dl, col in zip(DEADLINE_SCENARIOS, colors_dl):
        fig.add_vline(x=dl, line_dash="dot", line_color=col,
                      annotation_text=f"Deadline {dl} bln")
    fig.update_layout(
        title='Estimasi Jadwal Rata-rata (Gantt Chart)',
        xaxis_title='Bulan ke-',
        barmode='stack',
        showlegend=False,
        height=400
    )
    return fig


def plot_resource_impact(results, simulator, scenarios):
    """Analisis dampak penambahan resource"""
    baseline_mean = float(results['Total_Duration'].mean())
    impact_data = []

    for s in scenarios:
        stage = s['stage']
        factor = 1 - s['reduction_factor']
        temp = results.copy()
        temp[stage] = temp[stage] * factor

        # recalculate total with dependencies
        totals = []
        for idx in range(len(temp)):
            times = {}
            for sname in simulator.stages.keys():
                deps = simulator.stages[sname].dependencies
                start = 0 if not deps else max(times.get(d, 0) for d in deps)
                dur = temp.loc[idx, sname] if sname == stage else results.loc[idx, sname]
                times[sname] = start + dur
            totals.append(max(times.values()))

        opt_mean = float(np.mean(totals))
        reduction = baseline_mean - opt_mean
        probs = {dl: float(np.mean(np.array(totals) <= dl)) for dl in DEADLINE_SCENARIOS}
        impact_data.append({
            'label': s['label'],
            'stage': STAGE_LABELS.get(stage, stage),
            'reduction': reduction,
            'opt_mean': opt_mean,
            'probs': probs
        })

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Pengurangan Durasi per Skenario (Bulan)',
                                        'Probabilitas Selesai per Deadline'))
    colors = px.colors.qualitative.Set1

    # Left: duration reduction
    for i, d in enumerate(impact_data):
        fig.add_trace(go.Bar(
            x=[d['reduction']], y=[d['label']],
            orientation='h',
            marker_color=colors[i % len(colors)],
            text=[f"{d['reduction']:.2f} bln"],
            textposition='auto',
            name=d['label']
        ), row=1, col=1)

    # Right: probability improvement
    dl_labels = [f'{dl} bln' for dl in DEADLINE_SCENARIOS]
    baseline_probs = [float(np.mean(results['Total_Duration'] <= dl)) for dl in DEADLINE_SCENARIOS]

    fig.add_trace(go.Bar(
        name='Baseline',
        x=dl_labels,
        y=baseline_probs,
        marker_color='#9CA3AF'
    ), row=1, col=2)

    for i, d in enumerate(impact_data):
        probs_list = [d['probs'][dl] for dl in DEADLINE_SCENARIOS]
        fig.add_trace(go.Bar(
            name=d['label'],
            x=dl_labels,
            y=probs_list,
            marker_color=colors[i % len(colors)],
            opacity=0.75
        ), row=1, col=2)

    fig.update_yaxes(title_text='Probabilitas', row=1, col=2)
    fig.update_layout(height=420, barmode='group',
                      title='Analisis Dampak Penambahan Resource')
    return fig, impact_data


# ============================================================================
# 5. SKENARIO RESOURCE
# ============================================================================
RESOURCE_SCENARIOS = [
    {
        'label': 'Tambah Pekerja Struktur',
        'stage': 'Struktur_Beton',
        'reduction_factor': 0.22,
        'desc': 'Tambah 10 pekerja khusus bekisting & besi → estimasi -22% durasi struktur'
    },
    {
        'label': 'Alat Berat Tambahan',
        'stage': 'Pondasi',
        'reduction_factor': 0.28,
        'desc': 'Tambah 2 unit excavator & crane → estimasi -28% durasi pondasi'
    },
    {
        'label': 'Insinyur MEP Tambahan',
        'stage': 'Instalasi_MEP',
        'reduction_factor': 0.25,
        'desc': 'Tambah 3 insinyur MEP berpengalaman → estimasi -25% durasi instalasi'
    },
    {
        'label': 'Tim Finishing Tambahan',
        'stage': 'Interior_dan_Finishing',
        'reduction_factor': 0.30,
        'desc': 'Tambah tim interior khusus lab → estimasi -30% durasi finishing'
    },
]


# ============================================================================
# 6. APLIKASI UTAMA
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">🏗️ Simulasi Monte Carlo<br>Estimasi Waktu Pembangunan Gedung FITE</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Studi Kasus 2.1 — [11S1221] Pemodelan dan Simulasi</b><br>
    Proyek pembangunan Gedung Fakultas Informatika & Teknik Elektro (FITE) 5 lantai dengan fasilitas lengkap:
    ruang kelas, laboratorium komputer, elektro, mobile, VR/AR, game, ruang dosen, toilet, dan ruang serbaguna.<br><br>
    Simulasi ini menggunakan <b>Metode Monte Carlo</b> dengan distribusi triangular untuk memodelkan
    ketidakpastian durasi setiap tahapan konstruksi.
    </div>
    """, unsafe_allow_html=True)

    # ---- SIDEBAR ----
    st.sidebar.markdown("## ⚙️ Konfigurasi Simulasi")
    num_sim = st.sidebar.slider(
        'Jumlah Iterasi Simulasi:',
        min_value=5000, max_value=50000, value=20000, step=1000,
        help='Lebih banyak iterasi = lebih akurat, lebih lama'
    )

    st.sidebar.markdown("### 📋 Parameter Tahapan (Bulan)")
    user_config = {}
    for stage_name, config in DEFAULT_STAGES_CONFIG.items():
        label = STAGE_LABELS.get(stage_name, stage_name)
        base = config['base_params']
        with st.sidebar.expander(f"🔧 {label}"):
            opt = st.number_input("Optimistic", min_value=0.5, max_value=24.0,
                                  value=float(base['optimistic']), step=0.5, key=f"opt_{stage_name}")
            ml = st.number_input("Most Likely", min_value=0.5, max_value=24.0,
                                 value=float(base['most_likely']), step=0.5, key=f"ml_{stage_name}")
            pes = st.number_input("Pessimistic", min_value=0.5, max_value=36.0,
                                  value=float(base['pessimistic']), step=0.5, key=f"pes_{stage_name}")
        new_config = dict(config)
        new_config['base_params'] = {'optimistic': opt, 'most_likely': ml, 'pessimistic': pes}
        user_config[stage_name] = new_config

    run_btn = st.sidebar.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.8rem;color:#555;">
    <b>Skenario Deadline:</b><br>
    🟢 16 bulan — Agresif<br>
    🟡 20 bulan — Realistis<br>
    🔴 24 bulan — Konservatif
    </div>
    """, unsafe_allow_html=True)

    # ---- SESSION STATE ----
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None

    if run_btn:
        with st.spinner('Menjalankan simulasi Monte Carlo... Harap tunggu...'):
            sim = MonteCarloConstructionSim(user_config, num_simulations=num_sim)
            res = sim.run_simulation()
            st.session_state.results = res
            st.session_state.simulator = sim
        st.success(f'✅ Simulasi selesai! {num_sim:,} iterasi berhasil dijalankan.')

    if st.session_state.results is None:
        st.markdown("""
        <div style="text-align:center;padding:3rem;background:#F0F9FF;border-radius:12px;">
        <h3>🚀 Siap memulai?</h3>
        <p>Atur parameter di sidebar, lalu klik <b>Jalankan Simulasi</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Konfigurasi Tahapan Aktif")
        for s, c in DEFAULT_STAGES_CONFIG.items():
            b = c['base_params']
            st.markdown(f"""
            <div class="stage-card">
            <b>{STAGE_LABELS.get(s, s)}</b> &nbsp;|&nbsp;
            Optimistic: {b['optimistic']} bln &nbsp;|&nbsp;
            Most Likely: {b['most_likely']} bln &nbsp;|&nbsp;
            Pessimistic: {b['pessimistic']} bln
            </div>
            """, unsafe_allow_html=True)
        return

    # ---- HASIL SIMULASI ----
    results = st.session_state.results
    simulator = st.session_state.simulator
    total = results['Total_Duration']
    mean_d = float(total.mean())
    median_d = float(np.median(total))
    std_d = float(total.std())
    ci80 = np.percentile(total, [10, 90])
    ci95 = np.percentile(total, [2.5, 97.5])

    # ---- METRIK UTAMA ----
    st.markdown('<h2 class="sub-header">📈 Statistik Utama Proyek</h2>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{mean_d:.1f}</h2><p>Rata-rata (Bulan)</p></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{median_d:.1f}</h2><p>Median (Bulan)</p></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{ci80[0]:.1f}–{ci80[1]:.1f}</h2><p>80% CI (Bulan)</p></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h2>{ci95[0]:.1f}–{ci95[1]:.1f}</h2><p>95% CI (Bulan)</p></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- TABS VISUALISASI ----
    st.markdown('<h2 class="sub-header">📊 Visualisasi Hasil Simulasi</h2>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribusi Durasi",
        "🎯 Probabilitas Penyelesaian",
        "🔍 Critical Path & Tahapan",
        "📊 Analisis Risiko",
        "⚡ Dampak Penambahan Resource"
    ])

    with tab1:
        fig_dist, stats = plot_distribution(results)
        st.plotly_chart(fig_dist, use_container_width=True)
        with st.expander("📋 Detail Statistik"):
            d1, d2 = st.columns(2)
            with d1:
                st.write("**Statistik Deskriptif:**")
                st.write(f"- Rata-rata: **{stats['mean']:.2f} bulan**")
                st.write(f"- Median: **{stats['median']:.2f} bulan**")
                st.write(f"- Standar Deviasi: **{stats['std']:.2f} bulan**")
                st.write(f"- Minimum: {stats['min']:.2f} bulan")
                st.write(f"- Maksimum: {stats['max']:.2f} bulan")
            with d2:
                st.write("**Confidence Intervals:**")
                st.write(f"- 80% CI: [{stats['ci80'][0]:.2f}, {stats['ci80'][1]:.2f}] bulan")
                st.write(f"- 95% CI: [{stats['ci95'][0]:.2f}, {stats['ci95'][1]:.2f}] bulan")
                buf_80 = float(np.percentile(total, 80)) - mean_d
                buf_95 = float(np.percentile(total, 95)) - mean_d
                st.write(f"- Safety Buffer (80%): +{buf_80:.2f} bulan")
                st.write(f"- Contingency Reserve (95%): +{buf_95:.2f} bulan")

    with tab2:
        fig_prob = plot_completion_prob(results)
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("**Analisis Probabilitas per Skenario Deadline:**")
        cols = st.columns(3)
        labels_dl = ['🟢 16 Bulan (Agresif)', '🟡 20 Bulan (Realistis)', '🔴 24 Bulan (Konservatif)']
        for i, (dl, lbl) in enumerate(zip(DEADLINE_SCENARIOS, labels_dl)):
            prob_on = float(np.mean(total <= dl))
            prob_late = 1 - prob_on
            risk_days = max(0.0, float(np.percentile(total, 95)) - dl)
            with cols[i]:
                st.metric(label=lbl, value=f"{prob_on:.1%}",
                          delta=f"{prob_late:.1%} risiko terlambat",
                          delta_color="inverse")
                st.caption(f"Potensi keterlambatan (P95): +{risk_days:.1f} bln")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            critical_df = simulator.critical_path_prob()
            fig_cp = plot_critical_path(critical_df)
            st.plotly_chart(fig_cp, use_container_width=True)
        with c2:
            fig_box = plot_boxplot(results, simulator.stages)
            st.plotly_chart(fig_box, use_container_width=True)

        st.plotly_chart(plot_gantt_mean(results, simulator.stages), use_container_width=True)

        with st.expander("🔍 Detail Data Critical Path"):
            cp_display = critical_df.copy()
            cp_display.index = [STAGE_LABELS.get(i, i) for i in cp_display.index]
            cp_display.columns = ['Prob. Critical', 'Korelasi dg Total', 'Rata-rata Durasi (bln)']
            st.dataframe(cp_display.sort_values('Prob. Critical', ascending=False)
                         .style.format({'Prob. Critical': '{:.1%}',
                                        'Korelasi dg Total': '{:.2f}',
                                        'Rata-rata Durasi (bln)': '{:.2f}'}),
                         use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        risk_df = simulator.risk_contribution()
        with c1:
            fig_risk = plot_risk_contribution(risk_df)
            st.plotly_chart(fig_risk, use_container_width=True)
        with c2:
            # Correlation heatmap
            corr_mat = results[list(simulator.stages.keys())].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_mat.values,
                x=[STAGE_LABELS.get(c, c) for c in corr_mat.columns],
                y=[STAGE_LABELS.get(c, c) for c in corr_mat.index],
                colorscale='RdBu', zmid=0,
                text=np.round(corr_mat.values, 2),
                texttemplate='%{text}', textfont={"size": 9}
            ))
            fig_corr.update_layout(title='Matriks Korelasi Antar Tahapan', height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("📋 Detail Kontribusi Risiko"):
            risk_display = risk_df.copy()
            risk_display.index = [STAGE_LABELS.get(i, i) for i in risk_display.index]
            risk_display.columns = ['Varians', 'Kontribusi (%)', 'Std Dev']
            st.dataframe(risk_display.sort_values('Kontribusi (%)', ascending=False)
                         .style.format({'Varians': '{:.3f}',
                                        'Kontribusi (%)': '{:.1f}%',
                                        'Std Dev': '{:.3f}'}),
                         use_container_width=True)

    with tab5:
        st.markdown("""
        <div class="info-box">
        Simulasi ini menghitung dampak penambahan resource (pekerja khusus, alat berat, insinyur) 
        terhadap pengurangan durasi tahapan dan probabilitas penyelesaian tepat waktu.
        </div>
        """, unsafe_allow_html=True)

        fig_res, impact_data = plot_resource_impact(results, simulator, RESOURCE_SCENARIOS)
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("**Ringkasan Dampak per Skenario Resource:**")
        res_table = []
        baseline_mean = float(results['Total_Duration'].mean())
        for s_cfg, impact in zip(RESOURCE_SCENARIOS, impact_data):
            row = {
                'Skenario': impact['label'],
                'Deskripsi': s_cfg['desc'],
                'Durasi Setelah (bln)': round(impact['opt_mean'], 2),
                'Pengurangan (bln)': round(impact['reduction'], 2),
                'Prob 16 bln': f"{impact['probs'][16]:.1%}",
                'Prob 20 bln': f"{impact['probs'][20]:.1%}",
                'Prob 24 bln': f"{impact['probs'][24]:.1%}",
            }
            res_table.append(row)
        df_res = pd.DataFrame(res_table)
        st.dataframe(df_res, use_container_width=True)

        best = max(impact_data, key=lambda x: x['reduction'])
        st.markdown(f"""
        <div class="warning-box">
        🏆 <b>Rekomendasi:</b> Skenario <b>{best['label']}</b> memberikan 
        pengurangan durasi terbesar ({best['reduction']:.2f} bulan), 
        dengan estimasi total proyek <b>{best['opt_mean']:.1f} bulan</b>.
        </div>
        """, unsafe_allow_html=True)

    # ---- REKOMENDASI MANAJEMEN RISIKO ----
    st.markdown('<h2 class="sub-header">🎯 Analisis Deadline & Rekomendasi</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        target = st.number_input("Masukkan deadline target (bulan):",
                                 min_value=10, max_value=36, value=20, step=1)
        prob_t = float(np.mean(total <= target))
        risk_t = max(0.0, float(np.percentile(total, 95)) - target)
        st.metric(
            label=f"Probabilitas selesai dalam {target} bulan",
            value=f"{prob_t:.1%}",
            delta=f"Potensi keterlambatan P95: {risk_t:.1f} bln" if risk_t > 0 else "✅ Aman",
            delta_color="inverse"
        )

    with c2:
        buf_80 = float(np.percentile(total, 80)) - mean_d
        buf_95 = float(np.percentile(total, 95)) - mean_d
        st.markdown(f"""
        <div class="info-box">
        <h4>🏗️ Rekomendasi Manajemen Risiko:</h4>
        • <b>Safety Buffer</b> (80% confidence): <b>+{buf_80:.1f} bulan</b><br>
        • <b>Contingency Reserve</b> (95% confidence): <b>+{buf_95:.1f} bulan</b><br><br>
        • <b>Estimasi jadwal direkomendasikan:</b><br>
          &nbsp;&nbsp;{mean_d:.1f} + {buf_80:.1f} = <b>{mean_d + buf_80:.1f} bulan</b> (80% CI)<br>
          &nbsp;&nbsp;{mean_d:.1f} + {buf_95:.1f} = <b>{mean_d + buf_95:.1f} bulan</b> (95% CI)
        </div>
        """, unsafe_allow_html=True)

    # ---- INFO TEKNIS ----
    with st.expander("ℹ️ Informasi Teknis Simulasi"):
        st.write(f"**Jumlah iterasi:** {num_sim:,}")
        st.write(f"**Jumlah tahapan:** {len(simulator.stages)}")
        st.write("**Distribusi digunakan:** Triangular (tiga titik estimasi)")
        st.write("**Faktor risiko:** Diskrit (bernoulli) dan Kontinu (normal)")
        st.write("**Seed:** 42 (reprodusibel)")

    # ---- FOOTER ----
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#888;font-size:0.85rem;">
    <b>Simulasi Monte Carlo — Estimasi Waktu Pembangunan Gedung FITE</b><br>
    [11S1221] Pemodelan dan Simulasi | Modul Praktikum 5<br>
    ⚠️ Hasil simulasi merupakan estimasi probabilistik, bukan prediksi pasti.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()