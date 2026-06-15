"""
pid-lab 服务端 — FastAPI + WebSocket仿真引擎 + 数据管理
"""
import asyncio
import csv
import io
import json
import os
import sqlite3
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from controller import (PIDController, PIDParams, CascadeController,
                        FeedforwardController, SmithPredictor)
from models import ProcessModel, ModelParams, identify_from_csv, identify_transfer_function
from autotune import AutoTuner
from analysis import compute_bode, compute_nyquist

app = FastAPI(title="pid-lab")
BASE_DIR = os.path.dirname(__file__)
PRESETS_DIR = os.path.join(BASE_DIR, "presets")
DB_PATH = os.path.join(BASE_DIR, "experiments.db")
os.makedirs(PRESETS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# ===== SQLite数据管理 =====

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        pid_params TEXT,
        model_params TEXT,
        sp REAL,
        description TEXT,
        data_json TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS presets (
        name TEXT PRIMARY KEY,
        params_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

init_db()


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


# ===== REST API: 实验数据管理 =====

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """上传CSV实验数据，返回辨识结果"""
    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 3:
        return {"error": "数据太少"}
    # 尝试解析: 假设第一行是表头，后面是数据
    try:
        header = rows[0]
        data = np.array([[float(x) for x in row] for row in rows[1:]])
    except ValueError:
        return {"error": "数据格式错误，请确保是数值型CSV"}

    if data.shape[1] < 2:
        return {"error": "至少需要2列: 时间和输出"}

    t = data[:, 0]
    if data.shape[1] >= 3:
        u = data[:, 1]
        y = data[:, 2]
    else:
        u = np.ones_like(t)  # 假设阶跃输入
        y = data[:, 1]

    result = identify_from_csv(t, u, y)
    return result


@app.post("/api/identify-tf")
async def identify_tf(data: dict):
    """从提供的数据辨识传递函数"""
    t = np.array(data.get("time", []))
    u = np.array(data.get("input", []))
    y = np.array(data.get("output", []))
    order = data.get("order", 2)
    if len(t) < 10:
        return {"error": "数据点太少"}
    return identify_transfer_function(t, y, u, order)


@app.get("/api/experiments")
async def list_experiments():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, created_at, description FROM experiments ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "created_at": r[2], "description": r[3]} for r in rows]


@app.post("/api/experiments")
async def save_experiment(data: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO experiments (name, created_at, pid_params, model_params, sp, description, data_json) VALUES (?,?,?,?,?,?,?)",
              (data.get("name", "未命名"), datetime.now().isoformat(),
               json.dumps(data.get("pid", {})), json.dumps(data.get("model", {})),
               data.get("sp", 50), data.get("description", ""),
               json.dumps(data.get("history", {}))))
    exp_id = c.lastrowid
    conn.commit()
    conn.close()
    return {"id": exp_id}


@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"error": "未找到"}
    return {
        "id": row[0], "name": row[1], "created_at": row[2],
        "pid": json.loads(row[3]), "model": json.loads(row[4]),
        "sp": row[5], "description": row[6],
        "history": json.loads(row[7]) if row[7] else {},
    }


@app.get("/api/export-report/{exp_id}")
async def export_report(exp_id: int):
    """导出实验报告(文本格式，可粘贴到论文)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"error": "未找到"}

    pid = json.loads(row[3])
    model = json.loads(row[4])
    history = json.loads(row[7]) if row[7] else {}

    report = f"""# PID控制实验报告
实验名称: {row[1]}
创建时间: {row[2]}

## 控制器参数
- Kp = {pid.get('Kp', 'N/A')}
- Ki = {pid.get('Ki', 'N/A')}
- Kd = {pid.get('Kd', 'N/A')}
- 算法形式: {pid.get('variant', 'parallel')}

## 被控对象
- 模型类型: {model.get('model_type', 'N/A')}
- 设定值: {row[5]}

## 仿真结果
- 数据点数: {len(history.get('t', []))}

## 原始数据
t,sp,pv,cv
"""
    for i in range(len(history.get('t', []))):
        report += f"{history['t'][i]},{history['sp'][i]},{history['pv'][i]},{history['cv'][i]}\n"

    return StreamingResponse(
        io.BytesIO(report.encode("utf-8")),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{exp_id}.md"}
    )


# ===== WebSocket =====

class SimulationEngine:
    def __init__(self):
        self.running = False
        self.paused = False
        self.pid_params = PIDParams()
        self.model_params = ModelParams()
        self.controller = PIDController(self.pid_params)
        self.model = ProcessModel(self.model_params)
        self.sp = 50.0
        self.t = 0.0
        self.dt = self.model_params.Ts
        self._last_cv = 0.0
        self.history = {"t": [], "sp": [], "pv": [], "cv": []}
        self.control_mode = "pid"  # pid / cascade / feedforward / smith

    def configure(self, pid_params, model_params, sp):
        model_changed = (model_params.model_type != self.model_params.model_type or
                         model_params.Ts != self.model_params.Ts or
                         model_params.K != self.model_params.K or
                         model_params.tau != self.model_params.tau or
                         model_params.theta != self.model_params.theta or
                         model_params.T != self.model_params.T or
                         model_params.zeta != self.model_params.zeta or
                         model_params.omega_n != self.model_params.omega_n or
                         model_params.tau1 != self.model_params.tau1 or
                         model_params.tau2 != self.model_params.tau2 or
                         model_params.zeta1 != self.model_params.zeta1 or
                         model_params.coupling != self.model_params.coupling or
                         model_params.custom_num != self.model_params.custom_num or
                         model_params.custom_den != self.model_params.custom_den)
        if model_changed:
            self.model_params = model_params
            self.model.set_params(model_params)
            self.model.reset()
            self.t = 0.0
            self.history = {"t": [], "sp": [], "pv": [], "cv": []}
        self.pid_params = pid_params
        self.controller.set_params(pid_params)
        self.sp = sp
        self.dt = model_params.Ts

    def reset(self):
        self.model.reset()
        self.controller.reset()
        self.t = 0.0
        self._last_cv = 0.0
        self.history = {"t": [], "sp": [], "pv": [], "cv": []}

    def step(self):
        prev_cv = self._last_cv
        pv = self.model.step(prev_cv, self.t)
        cv = self.controller.compute(self.sp, pv, self.dt)
        self._last_cv = cv
        self.history["t"].append(round(self.t, 4))
        self.history["sp"].append(self.sp)
        self.history["pv"].append(round(pv, 4))
        self.history["cv"].append(round(cv, 4))
        max_hist = 5000
        if len(self.history["t"]) > max_hist:
            for key in self.history:
                self.history[key] = self.history[key][-max_hist:]
        self.t += self.dt
        return pv, cv

    def compute_metrics(self):
        pv = self.history["pv"]
        sp = self.history["sp"]
        if len(pv) < 10:
            return {}
        sp_val = sp[-1]
        final_val = pv[-1]
        initial_val = pv[0] if pv[0] != 0 else 0.001
        peak_val = max(pv)
        overshoot = max(0, (peak_val - sp_val) / abs(sp_val) * 100) if sp_val != 0 else 0

        rise_low = initial_val + 0.1 * (sp_val - initial_val)
        rise_high = initial_val + 0.9 * (sp_val - initial_val)
        t10 = t90 = None
        for i, v in enumerate(pv):
            if t10 is None and v >= rise_low:
                t10 = self.history["t"][i]
            if t90 is None and v >= rise_high:
                t90 = self.history["t"][i]
        rise_time = (t90 - t10) if (t10 and t90) else 0

        band = 0.02 * abs(sp_val) if sp_val != 0 else 0.02
        settling_time = self.history["t"][-1]
        for i in range(len(pv) - 1, -1, -1):
            if abs(pv[i] - sp_val) > band:
                settling_time = self.history["t"][min(i + 1, len(pv) - 1)]
                break

        ss_error = sp_val - final_val
        iae = ise = 0.0
        dt = self.dt
        for i in range(1, len(pv)):
            e = sp[i] - pv[i]
            iae += abs(e) * dt
            ise += e * e * dt

        return {
            "overshoot": round(overshoot, 2),
            "settling_time": round(settling_time, 2),
            "rise_time": round(rise_time, 2),
            "ss_error": round(ss_error, 4),
            "IAE": round(iae, 2), "ISE": round(ise, 2),
        }

    def export_csv(self):
        lines = ["t,sp,pv,cv"]
        for i in range(len(self.history["t"])):
            lines.append(f"{self.history['t'][i]},{self.history['sp'][i]},"
                         f"{self.history['pv'][i]},{self.history['cv'][i]}")
        return "\n".join(lines)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    engine = SimulationEngine()
    task = None
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "configure":
                pid = PIDParams(**data.get("pid", {}))
                model = ModelParams(**data.get("model", {}))
                engine.configure(pid, model, data.get("sp", 50.0))

            elif msg_type == "start":
                if task and not task.done():
                    task.cancel()
                engine.running = True
                engine.paused = False
                task = asyncio.create_task(_run_sim(ws, engine))

            elif msg_type == "pause":
                engine.paused = not engine.paused

            elif msg_type == "reset":
                engine.reset()
                await ws.send_json({"type": "reset_done"})

            elif msg_type == "setpoint":
                engine.sp = data.get("value", 50.0)

            elif msg_type == "disturbance":
                engine.model.params.dist_amplitude = data.get("amplitude", 0.0)
                engine.model.params.dist_time = engine.t

            elif msg_type == "noise":
                engine.model.params.noise_amplitude = data.get("amplitude", 0.0)

            elif msg_type == "autotune":
                method = data.get("method", "step_response")
                await ws.send_json({"type": "autotune_progress", "status": "running", "method": method})
                tuner = AutoTuner(engine.model)
                try:
                    if method == "optimize":
                        result = tuner.optimize(
                            criterion=data.get("params", {}).get("criterion", "ISE"),
                            sp=engine.sp,
                            duration=data.get("params", {}).get("duration", 30.0),
                        )
                    else:
                        func = getattr(tuner, method)
                        result = func(**data.get("params", {}))
                    await ws.send_json({
                        "type": "autotune_result",
                        "Kp": round(result.Kp, 4), "Ki": round(result.Ki, 4),
                        "Kd": round(result.Kd, 4), "method": result.method,
                        "identification": result.identification,
                    })
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif msg_type == "freq_analysis":
                try:
                    bode_data = compute_bode(engine.pid_params, engine.model_params)
                    nyquist_data = compute_nyquist(engine.pid_params, engine.model_params)
                    await ws.send_json({
                        "type": "freq_data",
                        "bode": asdict(bode_data),
                        "nyquist": asdict(nyquist_data),
                    })
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif msg_type == "save_preset":
                name = data.get("name", "default")
                preset = {"pid": data.get("pid", {}), "model": data.get("model", {}),
                          "sp": data.get("sp", 50.0)}
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO presets (name, params_json, created_at) VALUES (?,?,?)",
                          (name, json.dumps(preset), datetime.now().isoformat()))
                conn.commit()
                conn.close()
                await ws.send_json({"type": "preset_saved", "name": name})

            elif msg_type == "load_preset":
                name = data.get("name", "default")
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT params_json FROM presets WHERE name=?", (name,))
                row = c.fetchone()
                conn.close()
                if row:
                    await ws.send_json({"type": "preset_loaded", "data": json.loads(row[0])})

            elif msg_type == "list_presets":
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT name FROM presets ORDER BY name")
                files = [r[0] for r in c.fetchall()]
                conn.close()
                await ws.send_json({"type": "preset_list", "presets": files})

            elif msg_type == "export_csv":
                csv_data = engine.export_csv()
                await ws.send_json({"type": "csv_data", "data": csv_data})

            elif msg_type == "save_experiment":
                name = data.get("name", f"实验_{int(time.time())%10000}")
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO experiments (name, created_at, pid_params, model_params, sp, description, data_json) VALUES (?,?,?,?,?,?,?)",
                          (name, datetime.now().isoformat(),
                           json.dumps(asdict(engine.pid_params)),
                           json.dumps(asdict(engine.model_params)),
                           engine.sp, data.get("description", ""),
                           json.dumps(engine.history)))
                exp_id = c.lastrowid
                conn.commit()
                conn.close()
                await ws.send_json({"type": "experiment_saved", "id": exp_id, "name": name})

    except WebSocketDisconnect:
        engine.running = False
        if task and not task.done():
            task.cancel()
    except Exception:
        engine.running = False
        if task and not task.done():
            task.cancel()


async def _run_sim(ws: WebSocket, engine: SimulationEngine):
    try:
        while engine.running:
            if not engine.paused:
                pv, cv = engine.step()
                metrics = engine.compute_metrics()
                try:
                    await ws.send_json({
                        "type": "sim_state",
                        "t": round(engine.t, 4), "sp": engine.sp,
                        "pv": round(pv, 4), "cv": round(cv, 4),
                        "metrics": metrics,
                    })
                except Exception:
                    break
            await asyncio.sleep(engine.dt)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
