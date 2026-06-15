"""
频域分析 — Bode图 + Nyquist图 + 增益/相位裕度
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import lti, bode, freqresp
from controller import PIDParams
from models import ModelParams, build_continuous_tf


@dataclass
class BodeResult:
    freq: list
    mag_db: list
    phase_deg: list
    gain_margin_db: float
    gain_margin_freq: float
    phase_margin_deg: float
    phase_margin_freq: float


@dataclass
class NyquistResult:
    real_pos: list
    imag_pos: list
    real_neg: list
    imag_neg: list


def pid_to_tf(params: PIDParams):
    """把PID参数转成连续传递函数 (num, den)"""
    Kp, Ki, Kd = params.Kp, params.Ki, params.Kd
    Ti, Td, Tf = params.Ti, params.Td, params.Tf

    if params.variant == "isa":
        # C(s) = Kp * (1 + 1/(Ti*s) + Td*s/(Tf*s+1))
        # 通分: 分子 = Kp*[(Ti*Tf+Ti*Td)*s^2 + (Ti+Tf)*s + 1]
        #        分母 = Ti*Tf*s^2 + Ti*s
        num = Kp * np.array([Ti * Tf + Ti * Td, Ti + Tf, 1.0])
        den = np.array([Ti * Tf, Ti, 0.0])
    else:
        # parallel / incremental 都用并行形式近似
        # C(s) = Kp + Ki/s + Kd*s/(Tf*s+1)
        # 分子 = (Kp*Tf+Kd)*s^2 + (Kp+Ki*Tf)*s + Ki
        # 分母 = Tf*s^2 + s
        num = np.array([Kp * Tf + Kd, Kp + Ki * Tf, Ki])
        den = np.array([Tf, 1.0, 0.0])

    return num, den


def compute_open_loop(pid_params: PIDParams, model_params: ModelParams):
    """开环传递函数 L(s) = C(s) * G(s)"""
    num_c, den_c = pid_to_tf(pid_params)
    # 频域分析需要包含延迟，用Pade近似
    num_g, den_g = build_continuous_tf(model_params.model_type, model_params, include_delay=True)
    num_ol = np.convolve(num_c, num_g)
    den_ol = np.convolve(den_c, den_g)
    return num_ol, den_ol


def compute_bode(pid_params: PIDParams, model_params: ModelParams,
                 n_points: int = 500) -> BodeResult:
    """计算Bode图 + 裕度"""
    num_ol, den_ol = compute_open_loop(pid_params, model_params)

    # 过滤掉den中的零系数（首项不能为零）
    while len(den_ol) > 1 and abs(den_ol[0]) < 1e-12:
        den_ol = den_ol[1:]
    while len(num_ol) > 1 and abs(num_ol[0]) < 1e-12:
        num_ol = num_ol[1:]

    try:
        sys = lti(num_ol, den_ol)
        w, mag_db, phase_deg = bode(sys, n=n_points)
    except Exception:
        # 退化情况返回空
        return BodeResult([], [], [], float('inf'), 0.0, float('inf'), 0.0)

    freq = w.tolist()
    mag = mag_db.tolist()
    phase = phase_deg.tolist()

    # 计算裕度
    gm_db, gm_freq = _gain_margin(w, mag_db, phase_deg)
    pm_deg, pm_freq = _phase_margin(w, mag_db, phase_deg)

    return BodeResult(
        freq=freq, mag_db=mag, phase_deg=phase,
        gain_margin_db=gm_db, gain_margin_freq=gm_freq,
        phase_margin_deg=pm_deg, phase_margin_freq=pm_freq
    )


def _gain_margin(w, mag_db, phase_deg):
    """增益裕度: 相位穿越-180°时的增益"""
    phase_rad = np.unwrap(np.radians(phase_deg))
    target = -np.pi

    for i in range(len(phase_rad) - 1):
        if (phase_rad[i] - target) * (phase_rad[i + 1] - target) < 0:
            # 线性插值
            frac = (target - phase_rad[i]) / (phase_rad[i + 1] - phase_rad[i])
            w_cross = w[i] + frac * (w[i + 1] - w[i])
            mag_at = np.interp(w_cross, w, mag_db)
            return float(-mag_at), float(w_cross)

    return float('inf'), 0.0


def _phase_margin(w, mag_db, phase_deg):
    """相位裕度: 增益穿越0dB时的相位裕量"""
    for i in range(len(mag_db) - 1):
        if mag_db[i] * mag_db[i + 1] < 0:
            frac = (0.0 - mag_db[i]) / (mag_db[i + 1] - mag_db[i])
            w_cross = w[i] + frac * (w[i + 1] - w[i])
            phase_at = np.interp(w_cross, w, phase_deg)
            return float(180.0 + phase_at), float(w_cross)

    return float('inf'), 0.0


def compute_nyquist(pid_params: PIDParams, model_params: ModelParams,
                    n_points: int = 500) -> NyquistResult:
    """计算Nyquist图"""
    num_ol, den_ol = compute_open_loop(pid_params, model_params)

    while len(den_ol) > 1 and abs(den_ol[0]) < 1e-12:
        den_ol = den_ol[1:]
    while len(num_ol) > 1 and abs(num_ol[0]) < 1e-12:
        num_ol = num_ol[1:]

    try:
        sys = lti(num_ol, den_ol)
        w_pos = np.logspace(-2, 2, n_points)
        _, H = freqresp(sys, w=w_pos)
    except Exception:
        return NyquistResult([], [], [], [])

    real_pos = np.real(H).tolist()
    imag_pos = np.imag(H).tolist()
    # 负频率是正频率的共轭
    real_neg = np.real(H)[::-1].tolist()
    imag_neg = (-np.imag(H)[::-1]).tolist()

    return NyquistResult(
        real_pos=real_pos, imag_pos=imag_pos,
        real_neg=real_neg, imag_neg=imag_neg
    )
