import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec

# ===================== 全局物理参数 =====================
omega1 = 1.0
omega2 = 1.0
F0 = 1.0
t_max = 50.0
t = np.linspace(0, t_max, 800)
steady_idx = int(0.6 * len(t))  # 定义为全局变量，绘图时可直接用

# 中文正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 耦合振动方程 =====================
def coupled_ode(t, y, beta1, beta2, k, omega):
    x1, v1, x2, v2 = y
    a1 = -2 * beta1 * v1 - (omega1**2 + k) * x1 + k * x2 + F0 * np.cos(omega * t)
    a2 = -2 * beta2 * v2 + k * x1 - (omega2**2 + k) * x2
    return [v1, a1, v2, a2]

# ===================== 求解计算 =====================
def simulate(beta1, beta2, k, omega):
    sol = solve_ivp(
        coupled_ode, [0, t_max], [0, 0, 0, 0],
        t_eval=t, args=(beta1, beta2, k, omega)
    )
    x1 = sol.y[0]
    x2 = sol.y[2]
    v1 = sol.y[1]
    v2 = sol.y[3]

    # 稳态数据（后40%）
    A1 = np.max(np.abs(x1[steady_idx:]))
    A2 = np.max(np.abs(x2[steady_idx:]))

    # 平均能量效率
    E1 = 0.5 * np.mean(v1[steady_idx:] ** 2)
    E2 = 0.5 * np.mean(v2[steady_idx:] ** 2)
    eta = (E2 / (E1 + E2 + 1e-6)) * 100

    # 实时能量平滑
    window = 50
    eta_real = (0.5 * v2**2) / (0.5*(v1**2 + v2**2) + 1e-6) * 100
    eta_real_smooth = np.convolve(np.ones(window)/window, eta_real, mode='same')

    # 触发条件（合理宽松，确保能看到现象）
    is_coupled = (A2 > A1 * 0.7)
    return x1, x2, A1, A2, eta, eta_real_smooth, is_coupled

# ===================== 页面布局（原版一模一样） =====================
st.set_page_config(page_title="耦合共振实验", layout="wide")
st.title("📊 多自由度耦合共振虚拟实验")
st.markdown("主振子受迫振动 + 从振子耦合共振 | 虚拟仿真实验")
st.divider()

# 侧边栏控制面板
with st.sidebar:
    st.header("🔧 实验参数调节")
    beta1 = st.slider("主振子阻尼 β₁", 0.05, 0.5, 0.10, 0.01)
    beta2 = st.slider("从振子阻尼 β₂", 0.05, 0.5, 0.10, 0.01)
    k     = st.slider("耦合强度 k", 0.1, 1.0, 0.50, 0.05)
    omega = st.slider("驱动力频率 ω", 0.5, 1.5, 1.00, 0.05)

# 运行计算
x1, x2, A1, A2, eta, eta_real_smooth, is_coupled = simulate(beta1, beta2, k, omega)
freq_ratio = omega / omega1

# ===================== 绘图区域（原版三图布局） =====================
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[7, 3], height_ratios=[1, 1])

# 1. 位移时域曲线
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, x1, 'r-', lw=1.8, label=f'主振子 x₁  A₁={A1:.2f}')
ax1.plot(t, x2, 'b-', lw=1.8, label=f'从振子 x₂  A₂={A2:.2f}')
ax1.axvline(t[steady_idx], c='gray', ls='--', label='稳态起点')
ax1.set_title("振动位移曲线")
ax1.legend()
ax1.grid(alpha=0.3)

# 2. 能量效率曲线
ax2 = fig.add_subplot(gs[1, 0])
ax2.axhline(50, color='red', ls='--', lw=1.2, label='共振阈值 50%')
ax2.plot(t, eta_real_smooth, 'g-', lw=1.8, label=f'平均效率 {eta:.1f}%')
ax2.set_ylim(0, 100)
ax2.set_title("能量传递效率")
ax2.legend()
ax2.grid(alpha=0.3)

# 3. 共振危险区域图
ax3 = fig.add_subplot(gs[:, 1])
fr = np.linspace(0.5, 1.5, 80)
kk = np.linspace(0.1, 1.0, 80)
FR, KK = np.meshgrid(fr, kk)
danger = (FR >= 0.8) & (FR <= 1.2) & (KK >= 0.3)
ax3.contourf(FR, KK, danger, cmap='Reds', alpha=0.4)
ax3.scatter(freq_ratio, k, c='blue', s=80, marker='*', label='当前工作点')
ax3.set_xlabel("频率比 ω/ω₁")
ax3.set_ylabel("耦合强度 k")
ax3.set_title("共振耦合危险区域")
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ===================== 结果面板（原版四列） =====================
st.divider()
st.subheader("📌 实验结果")
col1, col2, col3, col4 = st.columns(4)
col1.metric("频率比 ω/ω₁", f"{freq_ratio:.2f}")
col2.metric("主振子振幅 A₁", f"{A1:.2f}")
col3.metric("从振子振幅 A₂", f"{A2:.2f}")
col4.metric("能量效率 η", f"{eta:.1f}%")

st.divider()
# 共振状态显示
if is_coupled:
    st.error("⚠️ 共振耦合状态：已触发")
else:
    st.success("共振耦合状态：未触发")