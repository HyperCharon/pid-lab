"""
被控对象模型 — 支持内置模型 + 自定义传递函数 + 实测数据导入
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.signal import cont2discrete, lfilter, lfilter_zi


@dataclass
class ModelParams:
    model_type: str = "dc_motor"
    Ts: float = 0.1
    K: float = 1.0
    tau: float = 1.0
    theta: float = 0.5
    T: float = 2.0
    zeta: float = 0.7
    omega_n: float = 2.0
    tau1: float = 0.5
    tau2: float = 3.0
    zeta1: float = 0.15
    coupling: float = 0.3
    dist_time: float = 0.0
    dist_amplitude: float = 0.0
    noise_amplitude: float = 0.0
    # 自定义传递函数系数 (逗号分隔的字符串)
    custom_num: str = ""
    custom_den: str = ""


# ===== Pade近似 =====

def _pade_coefficients(order: int, theta: float):
    from math import comb
    n = order
    norm = comb(2 * n, n)
    num = np.zeros(n + 1)
    den = np.zeros(n + 1)
    for k in range(n + 1):
        c = comb(2 * n - k, n) / norm * (theta / 2.0) ** k
        num[k] = ((-1) ** k) * c
        den[k] = c
    return num, den


def _pade_delay(theta: float, order: int = 3):
    if theta <= 0:
        return np.array([1.0]), np.array([1.0])
    return _pade_coefficients(order, theta)


# ===== 模型构建 =====

def build_continuous_tf(model_type: str, params: ModelParams, include_delay: bool = False):
    """返回连续传递函数 (num, den)"""
    K, tau, theta, T = params.K, params.tau, params.theta, params.T
    zeta, omega_n = params.zeta, params.omega_n
    tau1, tau2, zeta1 = params.tau1, params.tau2, params.zeta1

    if model_type == "custom":
        # 用户自定义传递函数
        num = np.array([float(x) for x in params.custom_num.split(",") if x.strip()])
        den = np.array([float(x) for x in params.custom_den.split(",") if x.strip()])
        if len(num) == 0 or len(den) == 0:
            return np.array([1.0]), np.array([1.0, 1.0])
        return num, den

    elif model_type == "dc_motor":
        return np.array([K]), np.array([tau, 1.0])

    elif model_type == "temperature":
        num_g = np.array([K])
        den_g = np.array([T, 1.0])
        if include_delay and theta > 0:
            num_d, den_d = _pade_delay(theta, order=3)
            return np.convolve(num_g, num_d), np.convolve(den_g, den_d)
        return num_g, den_g

    elif model_type == "tank_level":
        return np.array([K]), np.array([T, 1.0, 0.0])

    elif model_type == "servo":
        num = np.array([K])
        den_2nd = np.array([tau1**2, 2.0 * zeta1 * tau1, 1.0])
        den_1st = np.array([tau2, 1.0])
        return num, np.convolve(den_2nd, den_1st)

    elif model_type == "second_order":
        return np.array([omega_n**2]), np.array([1.0, 2.0 * zeta * omega_n, omega_n**2])

    elif model_type == "coupled_tank":
        num1, den1 = np.array([K]), np.array([tau, 1.0])
        num2, den2 = np.array([params.coupling * K]), np.array([tau * 2.0, 1.0])
        num = np.convolve(num1, den2) + np.convolve(num2, den1)
        den = np.convolve(den1, den2)
        return num, den

    elif model_type == "integrating":
        # G(s) = K / s  (纯积分)
        return np.array([K]), np.array([1.0, 0.0])

    elif model_type == "sopdt":
        # 二阶加滞后: G(s) = K * e^(-theta*s) / ((tau1*s+1)*(tau2*s+1))
        num_g = np.array([K])
        den_g = np.convolve(np.array([tau1, 1.0]), np.array([tau2, 1.0]))
        if include_delay and theta > 0:
            num_d, den_d = _pade_delay(theta, order=3)
            return np.convolve(num_g, num_d), np.convolve(den_g, den_d)
        return num_g, den_g

    else:
        return np.array([K]), np.array([tau, 1.0])


class ProcessModel:
    def __init__(self, params: ModelParams):
        self.params = params
        self._build(params)

    def _build(self, params: ModelParams):
        num_c, den_c = build_continuous_tf(params.model_type, params)
        num_d, den_d, _dt = cont2discrete((num_c, den_c), params.Ts, method='zoh')
        self._b = num_d.flatten()
        self._a = den_d.flatten()
        try:
            zi = lfilter_zi(self._b, self._a)
            self._zi = zi * 0.0
        except np.linalg.LinAlgError:
            self._zi = np.zeros(max(len(self._b), len(self._a)) - 1)
        has_delay = (params.model_type in ("temperature", "sopdt") and params.theta > 0)
        self._delay_steps = round(params.theta / params.Ts) if has_delay else 0
        self._delay_buf = [0.0] * self._delay_steps if self._delay_steps > 0 else []

    def set_params(self, params: ModelParams):
        self.params = params
        self._build(params)

    def reset(self):
        try:
            zi = lfilter_zi(self._b, self._a)
            self._zi = zi * 0.0
        except np.linalg.LinAlgError:
            self._zi = np.zeros(max(len(self._b), len(self._a)) - 1)
        self._delay_buf = [0.0] * self._delay_steps if self._delay_steps > 0 else []
        self._t = 0.0

    def step(self, u: float, t: float) -> float:
        self._t = t
        y, self._zi = lfilter(self._b, self._a, [u], zi=self._zi)
        y_raw = y[0]
        if self._delay_steps > 0:
            self._delay_buf.append(y_raw)
            pv = self._delay_buf.pop(0)
        else:
            pv = y_raw
        if self.params.dist_amplitude != 0 and self.params.dist_time > 0:
            if t >= self.params.dist_time:
                pv += self.params.dist_amplitude
        if self.params.noise_amplitude > 0:
            pv += np.random.normal(0, self.params.noise_amplitude)
        return pv

    def step_response(self, duration: float = 50.0) -> tuple:
        """计算阶跃响应，返回 (time, output)"""
        steps = int(duration / self.params.Ts)
        t_hist, y_hist = [], []
        self.reset()
        for k in range(steps):
            t = k * self.params.Ts
            y = self.step(1.0, t)
            t_hist.append(t)
            y_hist.append(y)
        return np.array(t_hist), np.array(y_hist)


# ===== 系统辨识 =====

def identify_foptd(t: np.ndarray, y: np.ndarray, step_size: float = 1.0):
    """从阶跃响应数据辨识一阶加滞后模型 K, L, T"""
    y_ss = y[-1]
    K_gain = y_ss / step_size if step_size != 0 else 1.0
    dy = np.gradient(y, t)
    inf_idx = np.argmax(np.abs(dy))
    slope = dy[inf_idx]
    y_inf = y[inf_idx]
    t_inf = t[inf_idx]
    if abs(slope) > 1e-10:
        L = max(0.0, t_inf - y_inf / slope)
        T = max(0.01, y_ss / slope)
    else:
        L, T = 0.0, 1.0
    return K_gain, L, T


def identify_from_csv(time_col: np.ndarray, input_col: np.ndarray,
                      output_col: np.ndarray, model_order: str = "foptd"):
    """从实测CSV数据辨识传递函数模型

    参数:
        time_col: 时间列
        input_col: 输入(激励)列
        output_col: 输出(响应)列
        model_order: "foptd" (一阶+滞后) 或 "sopdt" (二阶+滞后)

    返回:
        dict: 辨识结果 {'K', 'L', 'T', 'num', 'den', 'fit_quality'}
    """
    # 归一化
    u_step = input_col[-1] - input_col[0] if len(input_col) > 0 else 1.0
    if abs(u_step) < 1e-10:
        u_step = 1.0
    y_norm = (output_col - output_col[0]) / u_step
    t = time_col - time_col[0]

    K, L, T = identify_foptd(t, y_norm, step_size=1.0)

    # 计算拟合质量 (R²)
    y_sim = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] < L:
            y_sim[i] = 0.0
        else:
            y_sim[i] = K * (1.0 - np.exp(-(t[i] - L) / T))
    ss_res = np.sum((y_norm - y_sim) ** 2)
    ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 构造传递函数
    if model_order == "foptd":
        num = np.array([K])
        den = np.array([T, 1.0])
    else:
        # 二阶: 用两个相同时间常数
        num = np.array([K])
        den = np.array([T**2, 2 * T, 1.0])

    return {
        "K": float(K), "L": float(L), "T": float(T),
        "num": num.tolist(), "den": den.tolist(),
        "fit_quality": float(r_squared),
        "time": t.tolist(), "output_norm": y_norm.tolist(),
        "output_fit": y_sim.tolist(),
    }


def identify_transfer_function(t: np.ndarray, y: np.ndarray, u: np.ndarray,
                                order: int = 2):
    """用ARX模型辨识任意阶传递函数

    参数:
        t: 时间数组
        y: 输出数组
        u: 输入数组
        order: 模型阶数

    返回:
        dict: {'num', 'den', 'fit_quality'}
    """
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    n = order

    # 构造ARX回归矩阵: y[k] = -a1*y[k-1] - ... + b0*u[k-1] + ...
    N = len(y) - n
    if N < 2 * n + 1:
        return {"num": [1.0], "den": [1.0, 1.0], "fit_quality": 0.0}

    Phi = np.zeros((N, 2 * n))
    Y = np.zeros(N)
    for i in range(N):
        k = i + n
        Y[i] = y[k]
        for j in range(n):
            Phi[i, j] = -y[k - j - 1]      # a系数
            Phi[i, n + j] = u[k - j - 1]    # b系数

    # 最小二乘
    try:
        theta_hat, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    except np.linalg.LinAlgError:
        return {"num": [1.0], "den": [1.0, 1.0], "fit_quality": 0.0}

    a_coeffs = [1.0] + theta_hat[:n].tolist()
    b_coeffs = theta_hat[n:].tolist()

    # 拟合质量
    y_sim = np.zeros_like(y)
    for k in range(n, len(y)):
        for j in range(n):
            y_sim[k] -= a_coeffs[j + 1] * y_sim[k - j - 1]
            y_sim[k] += b_coeffs[j] * u[k - j - 1] if k - j - 1 >= 0 else 0

    ss_res = np.sum((y - y_sim) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "num": b_coeffs,
        "den": a_coeffs,
        "fit_quality": float(r_squared),
    }
