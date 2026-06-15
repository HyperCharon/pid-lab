"""
PID控制器 — 三种算法 + 抗饱和 + 微分滤波 + 设定值加权 + 无扰切换
         + 串级控制 + 前馈控制 + Smith预估器
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDParams:
    Kp: float = 2.0
    Ki: float = 0.5
    Kd: float = 0.1
    Ti: float = 4.0
    Td: float = 0.2
    Tf: float = 0.02
    beta: float = 1.0
    gamma_: float = 1.0
    cv_min: float = 0.0
    cv_max: float = 100.0
    variant: str = "parallel"
    anti_windup: str = "clamping"
    back_calc_gain: float = 0.5


@dataclass
class PIDState:
    e_prev: float = 0.0
    e_prev2: float = 0.0
    integral: float = 0.0
    d_filtered: float = 0.0
    cv_prev: float = 0.0
    first_run: bool = True


class PIDController:
    def __init__(self, params: PIDParams):
        self.params = params
        self.state = PIDState()

    def reset(self):
        self.state = PIDState()

    def set_params(self, params: PIDParams):
        old_cv = self.state.cv_prev
        self.params = params
        self._bumpless_init(old_cv)

    def compute(self, sp: float, pv: float, dt: float) -> float:
        if dt <= 0:
            return self.state.cv_prev
        v = self.params.variant
        if v == "isa":
            return self._compute_isa(sp, pv, dt)
        elif v == "incremental":
            return self._compute_incremental(sp, pv, dt)
        else:
            return self._compute_parallel(sp, pv, dt)

    def _compute_parallel(self, sp, pv, dt):
        p, s = self.params, self.state
        e = sp - pv
        e_p = p.beta * sp - pv
        e_d = p.gamma_ * sp - pv
        P = p.Kp * e_p
        s.integral += (e + s.e_prev) / 2.0 * dt
        I = p.Ki * s.integral
        if s.first_run:
            d_raw, s.first_run = 0.0, False
        else:
            d_raw = (e_d - (p.gamma_ * sp - s.e_prev)) / dt
        alpha = p.Tf / (p.Tf + dt) if (p.Tf + dt) > 0 else 0.0
        s.d_filtered = alpha * s.d_filtered + (1.0 - alpha) * d_raw
        D = p.Kd * s.d_filtered
        cv_raw = P + I + D
        cv = self._apply_antiwindup(cv_raw, e, dt)
        s.e_prev = e
        s.cv_prev = cv
        return cv

    def _compute_isa(self, sp, pv, dt):
        p, s = self.params, self.state
        e = sp - pv
        e_p = p.beta * sp - pv
        e_d = p.gamma_ * sp - pv
        P = p.Kp * e_p
        s.integral += (e + s.e_prev) / 2.0 * dt
        I = (p.Kp / p.Ti) * s.integral if p.Ti > 0 else 0.0
        if s.first_run:
            d_raw, s.first_run = 0.0, False
        else:
            d_raw = (e_d - (p.gamma_ * sp - s.e_prev)) / dt
        alpha = p.Tf / (p.Tf + dt) if (p.Tf + dt) > 0 else 0.0
        s.d_filtered = alpha * s.d_filtered + (1.0 - alpha) * d_raw
        D = p.Kp * p.Td * s.d_filtered
        cv_raw = P + I + D
        cv = self._apply_antiwindup(cv_raw, e, dt)
        s.e_prev = e
        s.cv_prev = cv
        return cv

    def _compute_incremental(self, sp, pv, dt):
        p, s = self.params, self.state
        e = sp - pv
        delta_p = p.Kp * (e - s.e_prev)
        delta_i = p.Ki * e * dt
        if s.first_run:
            delta_d, s.first_run = 0.0, False
        else:
            delta_d = p.Kd * (e - 2.0 * s.e_prev + s.e_prev2) / dt
        delta_u = delta_p + delta_i + delta_d
        cv = max(p.cv_min, min(p.cv_max, s.cv_prev + delta_u))
        s.e_prev2 = s.e_prev
        s.e_prev = e
        s.cv_prev = cv
        return cv

    def _apply_antiwindup(self, cv_raw, e, dt):
        p, s = self.params, self.state
        if p.anti_windup == "clamping":
            cv = max(p.cv_min, min(p.cv_max, cv_raw))
            if (cv_raw >= p.cv_max and e > 0) or (cv_raw <= p.cv_min and e < 0):
                s.integral -= (e + s.e_prev) / 2.0 * dt
        else:
            cv = max(p.cv_min, min(p.cv_max, cv_raw))
            s.integral += p.back_calc_gain * (cv - cv_raw) * dt
        return cv

    def _bumpless_init(self, current_cv):
        p, s = self.params, self.state
        P = p.Kp * 0.0
        D = 0.0
        if p.variant == "isa":
            s.integral = (current_cv - P - D) * p.Ti / p.Kp if p.Kp > 0 and p.Ti > 0 else 0.0
        else:
            s.integral = (current_cv - P - D) / p.Ki if p.Ki > 0 else 0.0
        s.cv_prev = current_cv


# ===== 串级控制 =====

class CascadeController:
    """串级控制器: 外环(主控制器) + 内环(副控制器)"""
    def __init__(self, outer_params: PIDParams, inner_params: PIDParams):
        self.outer = PIDController(outer_params)
        self.inner = PIDController(inner_params)
        self._outer_cv = 0.0

    def reset(self):
        self.outer.reset()
        self.inner.reset()
        self._outer_cv = 0.0

    def compute(self, sp: float, pv_outer: float, pv_inner: float, dt: float) -> float:
        """sp=外环设定值, pv_outer=外环反馈, pv_inner=内环反馈"""
        self._outer_cv = self.outer.compute(sp, pv_outer, dt)
        cv = self.inner.compute(self._outer_cv, pv_inner, dt)
        return cv


# ===== 前馈+反馈控制 =====

class FeedforwardController:
    """前馈+反馈控制器"""
    def __init__(self, fb_params: PIDParams, ff_gain: float = 1.0):
        self.feedback = PIDController(fb_params)
        self.ff_gain = ff_gain
        self._ff_prev = 0.0

    def reset(self):
        self.feedback.reset()
        self._ff_prev = 0.0

    def compute(self, sp: float, pv: float, dt: float,
                disturbance: float = 0.0, d_disturbance: float = 0.0) -> float:
        """
        sp, pv: 反馈回路
        disturbance: 可测扰动
        d_disturbance: 扰动变化率
        """
        cv_fb = self.feedback.compute(sp, pv, dt)
        cv_ff = self.ff_gain * disturbance + self.ff_gain * 0.1 * d_disturbance
        return cv_fb + cv_ff


# ===== Smith预估器 =====

class SmithPredictor:
    """Smith预估器: 用于大滞后系统的控制"""
    def __init__(self, pid_params: PIDParams, model_K: float, model_T: float,
                 model_theta: float, Ts: float):
        self.controller = PIDController(pid_params)
        self.K = model_K
        self.T = model_T
        self.theta = model_theta
        self.Ts = Ts
        # 无滞后模型的状态
        self._model_nodelay_state = 0.0
        # 延迟缓冲
        self._delay_steps = max(0, round(model_theta / Ts))
        self._delay_buf = [0.0] * self._delay_steps if self._delay_steps > 0 else []
        self._model_delayed_state = 0.0

    def reset(self):
        self.controller.reset()
        self._model_nodelay_state = 0.0
        self._delay_buf = [0.0] * self._delay_steps if self._delay_steps > 0 else []
        self._model_delayed_state = 0.0

    def compute(self, sp: float, pv: float, u: float, dt: float) -> float:
        """
        sp: 设定值
        pv: 实际过程输出
        u: 上一步控制量
        """
        # 无滞后模型: x = K*(1-e^(-t/T))*u
        alpha = dt / (self.T + dt) if (self.T + dt) > 0 else 0.0
        self._model_nodelay_state += alpha * (self.K * u - self._model_nodelay_state)

        # 有滞后模型 (延迟线)
        if self._delay_steps > 0:
            self._delay_buf.append(self._model_nodelay_state)
            self._model_delayed_state = self._delay_buf.pop(0)
        else:
            self._model_delayed_state = self._model_nodelay_state

        # 修正反馈: pv_corrected = pv - model_delayed + model_nodelay
        pv_corrected = pv - self._model_delayed_state + self._model_nodelay_state

        return self.controller.compute(sp, pv_corrected, dt)
