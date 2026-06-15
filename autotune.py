"""
自动整定 — Z-N / Cohen-Coon / 继电反馈 / Lambda / 优化算法
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from models import ProcessModel, ModelParams, identify_foptd


@dataclass
class TuneResult:
    Kp: float
    Ki: float
    Kd: float
    method: str
    identification: dict


class AutoTuner:
    def __init__(self, model: ProcessModel):
        self.model = model

    def step_response(self, step_size: float = 1.0, duration: float = 50.0) -> TuneResult:
        """Ziegler-Nichols阶跃响应法"""
        dt = self.model.params.Ts
        steps = int(duration / dt)
        self.model.reset()
        t_hist, y_hist = [], []
        for k in range(steps):
            t = k * dt
            y = self.model.step(step_size, t)
            t_hist.append(t)
            y_hist.append(y)
        t_arr, y_arr = np.array(t_hist), np.array(y_hist)
        K_gain, L, T, ident = self._identify_foptd(t_arr, y_arr, step_size)

        if L > 0 and K_gain > 0:
            Kp = 1.2 * T / (K_gain * L)
            Ti = 2.0 * L
            Td = 0.5 * L
        else:
            Kp, Ti, Td = 0.5 / K_gain if K_gain > 0 else 1.0, T, 0.0
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td
        ident["method_name"] = "Ziegler-Nichols 阶跃响应"
        return TuneResult(Kp=Kp, Ki=Ki, Kd=Kd, method="step_response", identification=ident)

    def cohen_coon(self, step_size: float = 1.0, duration: float = 50.0) -> TuneResult:
        """Cohen-Coon整定法"""
        dt = self.model.params.Ts
        steps = int(duration / dt)
        self.model.reset()
        t_hist, y_hist = [], []
        for k in range(steps):
            t = k * dt
            y = self.model.step(step_size, t)
            t_hist.append(t)
            y_hist.append(y)
        t_arr, y_arr = np.array(t_hist), np.array(y_hist)
        K_gain, L, T, ident = self._identify_foptd(t_arr, y_arr, step_size)

        if L > 0 and K_gain > 0 and T > 0:
            a = K_gain * L / T
            Kp = (1.0 / (K_gain * a)) * (1.0 + a / 3.0)
            Ti = L * (32.0 + 6.0 * a) / (13.0 + 8.0 * a)
            Td = L * 4.0 / (11.0 + 2.0 * a)
        else:
            Kp, Ti, Td = 0.5 / K_gain if K_gain > 0 else 1.0, T, 0.0
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td
        ident["method_name"] = "Cohen-Coon"
        return TuneResult(Kp=Kp, Ki=Ki, Kd=Kd, method="cohen_coon", identification=ident)

    def relay_feedback(self, relay_amp: float = 5.0, duration: float = 30.0) -> TuneResult:
        """继电反馈法"""
        dt = self.model.params.Ts
        steps = int(duration / dt)
        self.model.reset()
        t_hist, pv_hist, u_hist = [], [], []
        pv = 0.0
        prev_u = relay_amp
        for k in range(steps):
            t = k * dt
            hysteresis = 0.01
            if pv > hysteresis:
                u = -relay_amp
            elif pv < -hysteresis:
                u = relay_amp
            else:
                u = prev_u
            prev_u = u
            pv = self.model.step(u, t)
            t_hist.append(t)
            pv_hist.append(pv)
            u_hist.append(u)

        t_arr = np.array(t_hist)
        pv_arr = np.array(pv_hist)
        u_arr = np.array(u_hist)
        half = len(pv_arr) // 2
        steady = pv_arr[half:]
        t_steady = t_arr[half:] - t_arr[half]

        zero_cross = []
        for i in range(1, len(steady)):
            if steady[i - 1] * steady[i] < 0:
                frac = -steady[i - 1] / (steady[i] - steady[i - 1])
                zero_cross.append(t_steady[i - 1] + frac * (t_steady[i] - t_steady[i - 1]))

        Tu = 2.0 * np.mean(np.diff(zero_cross)) if len(zero_cross) >= 2 else duration / 3.0
        a = (np.max(steady) - np.min(steady)) / 2.0
        Ku = 4.0 * relay_amp / (np.pi * a) if a > 0 else 1.0

        Kp = 0.6 * Ku
        Ti = 0.5 * Tu
        Td = 0.125 * Tu
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td

        peaks = []
        for i in range(1, len(steady) - 1):
            if (steady[i] > steady[i - 1] and steady[i] > steady[i + 1]) or \
               (steady[i] < steady[i - 1] and steady[i] < steady[i + 1]):
                peaks.append((float(t_steady[i]), float(steady[i])))

        ident = {
            "time": t_arr.tolist(), "pv_response": pv_arr.tolist(),
            "relay_output": u_arr.tolist(), "Tu": float(Tu), "Ku": float(Ku),
            "oscillation_amplitude": float(a),
            "zero_crossings": [(float(t), 0.0) for t in zero_cross],
            "peak_points": peaks, "steady_start": float(t_arr[half]),
            "method_name": "继电反馈法",
        }
        return TuneResult(Kp=Kp, Ki=Ki, Kd=Kd, method="relay_feedback", identification=ident)

    def lambda_tuning(self, lambda_c: float = 2.0, step_size: float = 1.0,
                      duration: float = 50.0) -> TuneResult:
        """Lambda整定法"""
        dt = self.model.params.Ts
        steps = int(duration / dt)
        self.model.reset()
        t_hist, y_hist = [], []
        for k in range(steps):
            t = k * dt
            y = self.model.step(step_size, t)
            t_hist.append(t)
            y_hist.append(y)
        t_arr, y_arr = np.array(t_hist), np.array(y_hist)
        K_gain, L, T, ident = self._identify_foptd(t_arr, y_arr, step_size)

        if K_gain > 0:
            Kp = T / (K_gain * (lambda_c + L))
            Ti = T
        else:
            Kp, Ti = 1.0, 1.0
        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = 0.0

        t_cl = np.linspace(0, duration, steps)
        desired_cl = 1.0 - np.exp(-t_cl / lambda_c) if lambda_c > 0 else np.ones_like(t_cl)
        ident["desired_response"] = desired_cl.tolist()
        ident["desired_time"] = t_cl.tolist()
        ident["lambda_c"] = float(lambda_c)
        ident["method_name"] = f"Lambda整定 (λ={lambda_c:.1f}s)"
        return TuneResult(Kp=Kp, Ki=Ki, Kd=Kd, method="lambda_tuning", identification=ident)

    def optimize(self, criterion: str = "ISE", sp: float = 50.0,
                 duration: float = 30.0, bounds: tuple = None) -> TuneResult:
        """用优化算法找最优PID参数

        criterion: "ISE" / "IAE" / "ITAE" / "overshoot" / "settling_time"
        """
        dt = self.model.params.Ts
        steps = int(duration / dt)

        def sim_with_params(pid_vec):
            Kp, Ki, Kd = pid_vec
            if Kp < 0 or Ki < 0 or Kd < 0:
                return 1e6
            from controller import PIDController, PIDParams
            ctrl = PIDController(PIDParams(Kp=Kp, Ki=Ki, Kd=Kd))
            model = ProcessModel(self.model.params)
            pv = 0.0
            errors = []
            for k in range(steps):
                cv = ctrl.compute(sp, pv, dt)
                pv = model.step(cv, k * dt)
                errors.append(sp - pv)

            e = np.array(errors)
            t = np.arange(steps) * dt

            if criterion == "ISE":
                return float(np.sum(e ** 2) * dt)
            elif criterion == "IAE":
                return float(np.sum(np.abs(e)) * dt)
            elif criterion == "ITAE":
                return float(np.sum(t * np.abs(e)) * dt)
            elif criterion == "overshoot":
                peak = np.max(np.abs(e))
                return float(peak / abs(sp) * 100 if sp != 0 else peak)
            else:
                return float(np.sum(e ** 2) * dt)

        # 初始猜测: 用Z-N方法
        try:
            zn = self.step_response()
            x0 = [max(0.1, zn.Kp), max(0.01, zn.Ki), max(0.0, zn.Kd)]
        except Exception:
            x0 = [1.0, 0.5, 0.1]

        if bounds is None:
            bounds = [(0.01, 100), (0.0, 50), (0.0, 20)]

        result = minimize(sim_with_params, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 0.001, 'fatol': 0.1})

        Kp, Ki, Kd = result.x
        ident = {
            "criterion": criterion,
            "initial_guess": x0,
            "optimal_value": float(result.fun),
            "iterations": int(result.nit),
            "converged": bool(result.success),
            "method_name": f"参数优化 ({criterion})",
        }
        return TuneResult(Kp=Kp, Ki=Ki, Kd=Kd, method="optimize", identification=ident)

    def _identify_foptd(self, t, y, step_size):
        K_gain, L, T = identify_foptd(t, y, step_size)
        y_ss = y[-1]
        dy = np.gradient(y, t)
        inf_idx = np.argmax(np.abs(dy))
        slope = dy[inf_idx]
        t_inf = t[inf_idx]
        y_inf = y[inf_idx]
        if abs(slope) > 1e-10:
            t_tangent = np.array([L, L + T * 1.5])
            y_tangent = np.array([0.0, slope * (L + T * 1.5 - t_inf) + y_inf])
        else:
            t_tangent, y_tangent = np.array([0, 1]), np.array([0, 0])

        ident = {
            "time": t.tolist(), "response": y.tolist(),
            "tangent_x": t_tangent.tolist(), "tangent_y": y_tangent.tolist(),
            "K": float(K_gain), "L": float(L), "T": float(T),
            "inflection_point": (float(t_inf), float(y_inf)),
            "y_ss": float(y_ss), "step_size": float(step_size),
        }
        return K_gain, L, T, ident
