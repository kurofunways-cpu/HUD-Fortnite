"""HUD V8 — rastreamento híbrido YOLO + fluxo óptico com controle EXPERIMENTAL.

Não lê memória, não injeta código e não contorna anticheat. O Fortnite pode
ignorar os eventos de mouse. Use por sua conta e risco; pode violar as regras.
"""

from __future__ import annotations

import queue
import cv2
import numpy as np
import sys
import threading
import time

from movimento import calcular_passo, janela_fortnite_ativa, nome_janela_ativa, mover_relativo
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

APP_DIR = Path(__file__).resolve().parent
PREVIEW_W, PREVIEW_H = 960, 540
DARK, PANEL, TEXT, ACCENT = "#101622", "#1d2a3b", "#eaf1fb", "#4fc3f7"


def resize_for_model(frame, max_width: int = 960):
    """Retorna imagem BGR reduzida preservando proporção."""
    import cv2
    h, w = frame.shape[:2]
    scale = min(max_width / max(w, 1), 1.0)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    return frame


def parse_yolo_detections(result):
    """Extrai caixas + IDs do tracker da API Ultralytics."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coords = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(coords)
    output = []
    for index, (xyxy, conf, ident) in enumerate(zip(coords, confidences, ids)):
        output.append({
            "box": tuple(int(v) for v in xyxy),
            "confidence": float(conf),
            # Sem ID, ainda dá para desenhar caixa, mas não travar um alvo com segurança.
            "id": int(ident) if ident is not None else None,
        })
    return output


def choose_box(detections, x, y):
    """Seleciona a menor caixa contendo o clique para evitar caixas sobrepostas."""
    candidates = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        if x1 <= x <= x2 and y1 <= y <= y2 and det["id"] is not None:
            candidates.append(((x2-x1)*(y2-y1), det))
    return min(candidates, key=lambda t: t[0])[1] if candidates else None


class FluxoRapido:
    def __init__(self):
        self.gray = None
        self.points = None
        self.box = None
        self.frames = 0

    def reset(self):
        self.gray = None
        self.points = None
        self.box = None
        self.frames = 0

    @staticmethod
    def _clip_box(box, width, height):
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(width - 1, x1)))
        y1 = int(max(0, min(height - 1, y1)))
        x2 = int(max(x1 + 1, min(width, x2)))
        y2 = int(max(y1 + 1, min(height, y2)))
        return x1, y1, x2, y2

    def _features(self, gray):
        if self.box is None:
            return None
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = self._clip_box(self.box, width, height)
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 12:
            return None
        inset_x = min(int(bw * .15), 20)
        inset_y = min(int(bh * .12), 24)
        xa, xb = x1 + inset_x, x2 - inset_x
        ya, yb = y1 + inset_y, y2 - inset_y
        if xb - xa < 8 or yb - ya < 8:
            xa, xb, ya, yb = x1, x2, y1, y2
        roi = gray[ya:yb, xa:xb]
        corners = cv2.goodFeaturesToTrack(
            roi, maxCorners=65, qualityLevel=.013, minDistance=5, blockSize=5)
        if corners is None or len(corners) < 4:
            return None
        corners[:, 0, 0] += xa
        corners[:, 0, 1] += ya
        return corners.astype(np.float32)

    def seed(self, frame, box):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reset()
        self.box = self._clip_box(box, frame.shape[1], frame.shape[0])
        self.points = self._features(gray)
        self.gray = gray
        return self.points is not None

    def update(self, frame):
        if self.gray is None or self.points is None or self.box is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, .02))
        if next_points is None or status is None:
            self.reset()
            return None
        good = status.reshape(-1) == 1
        if errors is not None:
            good &= errors.reshape(-1) < 38
        new_pts = next_points.reshape(-1, 2)[good]
        old_pts = self.points.reshape(-1, 2)[good]
        if len(new_pts) < 4:
            self.reset()
            return None
        flows = new_pts - old_pts
        median = np.median(flows, axis=0)
        residual = np.linalg.norm(flows - median, axis=1)
        ok = residual < 9.0
        if int(ok.sum()) < 4:
            self.reset()
            return None
        median = np.median(flows[ok], axis=0)
        dx, dy = float(median[0]), float(median[1])
        width, height = frame.shape[1], frame.shape[0]
        if abs(dx) > width * .16 or abs(dy) > height * .16:
            self.reset()
            return None
        x1, y1, x2, y2 = self.box
        shifted = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        if (shifted[2] < 5 or shifted[0] > width - 5
                or shifted[3] < 5 or shifted[1] > height - 5):
            self.reset()
            return None
        self.box = self._clip_box(shifted, width, height)
        self.frames += 1
        self.gray = gray
        self.points = new_pts[ok].reshape(-1, 1, 2).astype(np.float32)
        if self.frames % 8 == 0 or len(self.points) < 9:
            fresh = self._features(gray)
            if fresh is not None:
                self.points = fresh
        return self.box


class HUD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HUD V8 | Rastreamento híbrido")
        self.root.geometry("1030x790+90+70")
        self.root.minsize(1000, 760)
        self.root.configure(bg=DARK)
        self.messages = queue.Queue(maxsize=10)
        self.frame_queue = queue.Queue(maxsize=1)
        self.key_commands = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.thread = None
        self.listener = None
        self.keys_down = set()
        self.source = tk.StringVar(value="tela")
        self.video_path = ""
        self.latest_frame = None
        self.latest_boxes = []
        self.preview_mapping = None
        self.selected_id = None
        self.following = False
        self.lost_frames = 0
        self.last_photo = None
        self.running = False
        self.drag_origin = None
        self.enable_mouse = tk.BooleanVar(value=True)
        self.only_fortnite = tk.BooleanVar(value=True)
        self.ignore_self = tk.BooleanVar(value=True)
        self.mouse_gain = tk.DoubleVar(value=0.26)
        self.smooth_target = None
        self.active_box = None
        self.last_target_box = None
        self.last_target_at = 0.0
        self.last_draw_at = 0.0
        self.fast_thread = None
        self.flow_lock = threading.Lock()
        self.flow_generation = 0
        self.flow_seed_box = None
        self.flow_queue = queue.Queue(maxsize=1)
        self.fast_last_at = 0.0
        self.yolo_last_confirmed = 0.0
        self.last_flow_seed_at = 0.0
        self.yolo_frames_seen = 0
        self.fast_frames_seen = 0
        self.metrics_started_at = time.monotonic()
        self.metrics = tk.StringVar(value="YOLO: — FPS | Fluxo: — FPS")
        self.control_info = tk.StringVar(value="Controle de câmera: desligado")
        self.last_move_at = 0.0
        self.mouse_failure_warned = False
        self.last_control_notice_at = 0.0
        self._build_ui()
        self._init_hotkeys()
        self.root.after(80, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _label(self, parent, text, font=("Segoe UI", 10), fg=TEXT, bg=None, **kw):
        return tk.Label(parent, text=text, font=font, fg=fg, bg=bg or DARK, **kw)

    def _button(self, parent, text, action, color=PANEL):
        return tk.Button(parent, text=text, command=action, bg=color, fg="white", relief="flat",
                         font=("Segoe UI", 10, "bold"), padx=12, pady=8,
                         activebackground="#314e69", activeforeground="white", cursor="hand2")

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#253650", height=50)
        header.pack(fill="x", padx=8, pady=(8, 0))
        header.pack_propagate(False)
        title = self._label(header, "  HUD V8  •  YOLO + fluxo óptico rápido", ("Segoe UI", 13, "bold"), bg="#253650")
        title.pack(side="left", pady=9)
        self._button(header, "— Minimizar", self.root.iconify).pack(side="right", padx=6, pady=4)
        self._button(header, "× Fechar", self.close, "#683041").pack(side="right", pady=4)
        for widget in (header, title):
            widget.bind("<Button-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_motion)

        row = tk.Frame(self.root, bg=DARK)
        row.pack(fill="x", padx=12, pady=10)
        tk.Radiobutton(row, text="Tela ao vivo", variable=self.source, value="tela", fg=TEXT,
                       bg=DARK, selectcolor=PANEL, activebackground=DARK, activeforeground=TEXT,
                       font=("Segoe UI", 10)).pack(side="left")
        tk.Radiobutton(row, text="Vídeo gravado", variable=self.source, value="video", fg=TEXT,
                       bg=DARK, selectcolor=PANEL, activebackground=DARK, activeforeground=TEXT,
                       font=("Segoe UI", 10)).pack(side="left", padx=(15, 0))
        self._button(row, "Escolher vídeo", self.select_video).pack(side="left", padx=8)
        self._button(row, "▶ Iniciar prévia", self.start, "#1b6658").pack(side="left", padx=8)
        self._button(row, "■ Parar", self.stop).pack(side="left")
        self._button(row, "P: Pausar / retomar", self.toggle_follow).pack(side="left", padx=8)

        self.info = tk.StringVar(value="Pronto. Escolha uma fonte e clique em Iniciar prévia.")
        self._label(self.root, "Acompanha automaticamente após iniciar a prévia. P pausa/retoma; ESC pausa.",
                    ("Segoe UI", 10), fg="#9ad3f6").pack(anchor="w", padx=17)
        settings = tk.Frame(self.root, bg=DARK)
        settings.pack(fill="x", padx=16, pady=(3, 7))
        tk.Checkbutton(settings, text="Mover câmera (experimental)", variable=self.enable_mouse,
                       fg=TEXT, bg=DARK, selectcolor=PANEL, activebackground=DARK,
                       activeforeground=TEXT).pack(side="left")
        tk.Checkbutton(settings, text="Só com Fortnite em foco", variable=self.only_fortnite,
                       fg=TEXT, bg=DARK, selectcolor=PANEL, activebackground=DARK,
                       activeforeground=TEXT).pack(side="left", padx=12)
        tk.Checkbutton(settings, text="Ignorar personagem central", variable=self.ignore_self,
                       fg=TEXT, bg=DARK, selectcolor=PANEL, activebackground=DARK,
                       activeforeground=TEXT).pack(side="left", padx=7)
        speed = tk.Frame(self.root, bg=DARK)
        speed.pack(fill="x", padx=16, pady=(0, 3))
        self._label(speed, "Velocidade da câmera:").pack(side="left")
        tk.Scale(speed, from_=0.05, to=0.65, resolution=0.01, orient="horizontal",
                 length=270, variable=self.mouse_gain, bg=DARK, fg=TEXT,
                 highlightthickness=0, troughcolor=PANEL).pack(side="left")
        self._button(speed, "Testar mouse no Windows", self.test_mouse).pack(side="left", padx=8)
        self._label(speed, "P pausa/retoma; ESC pausa.",
                    ("Segoe UI", 9), fg="#f8c58c").pack(side="left", padx=10)

        self.canvas = tk.Canvas(self.root, width=PREVIEW_W, height=PREVIEW_H,
                                highlightthickness=1, highlightbackground="#466079", bg="#090e16")
        self.canvas.pack(padx=12)
        # Sem seleção manual necessária
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                                text="A prévia aparece aqui depois de iniciar", fill="#87a0b7",
                                font=("Segoe UI", 16))
        self._label(self.root, "Status", ("Segoe UI", 10, "bold"), fg=ACCENT).pack(anchor="w", padx=16, pady=(10, 2))
        tk.Label(self.root, textvariable=self.metrics, fg="#a8c7da", bg=DARK,
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=16)
        tk.Label(self.root, textvariable=self.info, font=("Segoe UI", 10), fg=TEXT, bg=DARK,
                 anchor="w").pack(fill="x", padx=16)
        self._label(self.root, "O jogo pode ignorar o movimento do mouse. Não desative o anticheat ou o Smart App Control.",
                    ("Segoe UI", 9), fg="#9aafc4").pack(anchor="w", padx=16, pady=(4, 0))

    def _drag_start(self, event):
        self.drag_origin = (event.x_root, event.y_root, self.root.winfo_x(), self.root.winfo_y())

    def _drag_motion(self, event):
        if self.drag_origin:
            x0, y0, wx, wy = self.drag_origin
            self.root.geometry(f"+{wx + event.x_root - x0}+{wy + event.y_root - y0}")

    def _init_hotkeys(self):
        try:
            from pynput.keyboard import Listener, Key

            def key_name(key):
                if key == Key.esc:
                    return "esc"
                ch = getattr(key, "char", None)
                return str(ch).lower() if ch else None

            def on_press(key):
                name = key_name(key)
                if name in ("p", "esc") and name not in self.keys_down:
                    self.keys_down.add(name)
                    self.key_commands.put(name)

            def on_release(key):
                name = key_name(key)
                if name in self.keys_down:
                    self.keys_down.discard(name)

            self.listener = Listener(on_press=on_press, on_release=on_release)
            self.listener.daemon = True
            self.listener.start()
        except Exception as exc:
            self.info.set(f"Atalhos globais indisponíveis ({exc}). Use os botões da janela.")
            self.root.bind_all("<KeyPress-p>", lambda event: self.toggle_follow())
            self.root.bind_all("<Escape>", lambda event: self.release_target())

    def test_mouse(self):
        """Teste local do SendInput no Windows, sem Fortnite em foco."""
        self.release_target()
        if sys.platform != "win32":
            self.info.set("Teste indisponível: funciona apenas no Windows.")
            return
        ok = mover_relativo(70, 0)
        if ok:
            self.info.set(
                "Teste local: Windows aceitou +70 px de mouse. "
                "Veja se o ponteiro andou. Isso não garante movimento no Fortnite."
            )
        else:
            self.info.set(
                "Teste local falhou: Windows rejeitou o evento de mouse. "
                "Não tente desativar proteções de segurança."
            )

    def select_video(self):
        path = filedialog.askopenfilename(title="Escolha um vídeo", filetypes=[
            ("Vídeos", "*.mp4 *.mkv *.avi *.mov *.webm"), ("Todos os arquivos", "*.*")])
        if path:
            self.video_path = path
            self.source.set("video")
            self.info.set(f"Vídeo selecionado: {Path(path).name}")

    def _post(self, kind, payload):
        if kind == "frame":
            # A prévia não deve apagar erros ou mensagens de diagnóstico.
            try:
                while True:
                    self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                pass
            return
        try:
            self.messages.put_nowait((kind, payload))
        except queue.Full:
            pass

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            self.info.set("A captura já está em andamento. Use Parar antes de trocar a fonte.")
            return
        if self.source.get() == "video" and not self.video_path:
            self.select_video()
            if not self.video_path:
                return
        self.stop_event.clear()
        self.following = True
        self.selected_id = None
        self.smooth_target = None
        self.active_box = None
        self.last_target_box = None
        self.last_target_at = 0.0
        self.last_draw_at = 0.0
        self.latest_boxes = []
        self.fast_last_at = 0.0
        self.yolo_last_confirmed = 0.0
        self.last_flow_seed_at = 0.0
        with self.flow_lock:
            self.flow_generation += 1
            self.flow_seed_box = None
        self.flow_queue = queue.Queue(maxsize=1)
        self.running = True
        src, path = self.source.get(), self.video_path
        self.thread = threading.Thread(target=self._worker, args=(src, path), daemon=True)
        self.thread.start()
        if src == "tela":
            self.fast_thread = threading.Thread(target=self._fast_worker, daemon=True)
            self.fast_thread.start()
        self.info.set("Rastreamento híbrido ativo. Iniciando IA e fluxo óptico...")

    def _worker(self, source, path):
        capture = None
        try:
            import cv2
            import mss
            import numpy as np
            from ultralytics import YOLO
            model = YOLO("yolo11n.pt")
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
            self._post("message", f"Modelo pronto (dispositivo: {device}). Rastreamento automático ativo.")
            if source == "video":
                capture = cv2.VideoCapture(path)
                if not capture.isOpened():
                    raise RuntimeError("Não foi possível abrir o vídeo selecionado.")
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            with mss.mss() as screen:
                monitor = screen.monitors[1]
                while not self.stop_event.is_set():
                    start = time.monotonic()
                    if source == "video":
                        good, frame = capture.read()
                        if not good:
                            self._post("message", "Fim do vídeo. Clique em Iniciar prévia para reproduzir novamente.")
                            break
                    else:
                        shot = np.asarray(screen.grab(monitor))
                        frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                    frame = resize_for_model(frame)
                    results = model.track(frame, persist=True, tracker="botsort.yaml",
                                          classes=[0], conf=0.25, imgsz=512, verbose=False,
                                          device=device)
                    dets = parse_yolo_detections(results[0])
                    self._post("frame", (frame, dets))
                    # Limite de atualização sem prometer taxa de quadros específica.
                    if source == "video":
                        remain = min(0.1, 1.0 / max(1.0, fps)) - (time.monotonic() - start)
                    else:
                        remain = 0.075 - (time.monotonic() - start)
                    if remain > 0:
                        self.stop_event.wait(remain)
        except Exception as exc:
            self._post("error", f"Falha: {exc}")
        finally:
            if capture is not None:
                capture.release()
            self._post("done", None)

    def stop(self):
        self.stop_event.set()
        with self.flow_lock:
            self.flow_generation += 1
            self.flow_seed_box = None
        self.following = False
        self.selected_id = None
        self.smooth_target = None
        self.last_target_box = None
        self.active_box = None
        self.info.set("Parando prévia e movimento de mouse...")

    def toggle_follow(self):
        """P: pausa ou retoma o rastreamento (ligado por padrão após iniciar)."""
        if not self.running:
            self.info.set("Clique em Iniciar prévia antes de usar P.")
            return
        if self.following:
            self.release_target()
            return
        self.following = True
        self.selected_id = None
        self.smooth_target = None
        self.last_target_box = None
        self.active_box = None
        self.last_target_at = 0.0
        self.fast_last_at = 0.0
        with self.flow_lock:
            self.flow_generation += 1
            self.flow_seed_box = None
        self.mouse_failure_warned = False
        self.info.set("Rastreamento retomado: procurando personagem visível.")

    def release_target(self):
        """P ou ESC: pausa imediatamente e libera o alvo anterior."""
        self.following = False
        self.selected_id = None
        self.smooth_target = None
        self.last_target_box = None
        self.active_box = None
        self.fast_last_at = 0.0
        with self.flow_lock:
            self.flow_generation += 1
            self.flow_seed_box = None
        self.info.set("Pausado. Pressione P para retomar.")

    def pick_target(self, event):
        if self.preview_mapping is None:
            return
        scale, dx, dy = self.preview_mapping
        x, y = (event.x - dx) / scale, (event.y - dy) / scale
        boxes = self._filtered_detections(self.latest_boxes, self.latest_frame.shape if self.latest_frame is not None else None)
        det = choose_box(boxes, x, y)
        if det is None:
            self.info.set("Clique dentro do retângulo de um personagem com ID detectado.")
            return
        self.selected_id = det["id"]
        self.lost_frames = 0
        self.following = False
        self.info.set(f"Selecionado ID {self.selected_id}. Aperte P para acompanhar visualmente.")

    def _filtered_detections(self, detections, frame_shape):
        if not self.ignore_self.get() or frame_shape is None:
            return detections
        h, w = frame_shape[:2]
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = (x1 + x2) / 2
            # Heurística para o avatar em terceira pessoa, no primeiro plano.
            # Pode ocultar aliados centrais muito próximos.
            is_self = (x2-x1 > w*.11 and abs(cx-w/2) < w*.22
                       and y2 > h*.88 and y1 > h*.18)
            if not is_self:
                filtered.append(det)
        return filtered

    def _selecionar_automaticamente(self, detections, frame_shape):
        """Mantém um alvo por ID; se o ID mudar, tenta reassociar pela posição.

        Quando ele some, aguarda 0,45 s antes de buscar outra pessoa.
        Não identifica equipes e não garante identidade em oclusões.
        """
        h, w = frame_shape[:2]
        agora = time.monotonic()
        validos = [d for d in detections
                   if d["box"][2] > d["box"][0] and d["box"][3] > d["box"][1]]

        escolhido = None
        if self.last_target_box is not None:
            lx1, ly1, lx2, ly2 = self.last_target_box
            lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            largura, altura = lx2 - lx1, ly2 - ly1
            raio = max(52.0, min(w * 0.23, largura * 2.1))
            raio_id = max(raio * 1.6, w * 0.28)

            # O ID é preferido, mas rejeitamos saltos grandes e improváveis.
            if self.selected_id is not None:
                for det in validos:
                    if det["id"] != self.selected_id:
                        continue
                    x1, y1, x2, y2 = det["box"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5 <= raio_id:
                        escolhido = det
                        break

            # Recupera o alvo perto da última posição se o rastreador trocar ID.
            if escolhido is None:
                candidatos = []
                for det in validos:
                    x1, y1, x2, y2 = det["box"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
                    det_largura, det_altura = x2 - x1, y2 - y1
                    proporcao = min(altura, det_altura) / max(altura, det_altura)
                    if dist <= raio and proporcao >= 0.48:
                        candidatos.append((dist, -det["confidence"], det))
                if candidatos:
                    escolhido = min(candidatos, key=lambda item: item[:2])[2]

            if escolhido is None:
                if agora - self.last_target_at < 0.45:
                    return None
                self.selected_id = None
                self.last_target_box = None
                self.smooth_target = None

        if escolhido is None and validos:
            centro_x, centro_y = w / 2, h / 2
            distancia_max = 0.83 * ((centro_x ** 2 + centro_y ** 2) ** 0.5)
            candidatos = []
            for det in validos:
                x1, y1, x2, y2 = det["box"]
                cx, cy = (x1 + x2) / 2, y1 + (y2 - y1) * 0.46
                distancia = ((cx - centro_x) ** 2 + (cy - centro_y) ** 2) ** 0.5
                if distancia <= distancia_max:
                    candidatos.append((distancia, -det["confidence"], det))
            if candidatos:
                escolhido = min(candidatos, key=lambda item: item[:2])[2]

        if escolhido is not None:
            if self.selected_id != escolhido["id"]:
                self.smooth_target = None
            self.selected_id = escolhido["id"]
            self.last_target_box = escolhido["box"]
            self.last_target_at = agora
        return escolhido

    def _set_flow_seed(self, box):
        """Envia uma caixa do YOLO ao rastreador de fluxo óptico."""
        with self.flow_lock:
            self.flow_generation += 1
            self.flow_seed_box = box
        self.last_flow_seed_at = time.monotonic()
        self.fast_last_at = 0.0

    def _fast_worker(self):
        """Captura em alta frequência sem esperar as inferências do YOLO."""
        try:
            import cv2
            import mss
            import numpy as np
            tracker = FluxoRapido()
            previous_generation = -1
            with mss.mss() as screen:
                monitor = screen.monitors[1]
                while not self.stop_event.is_set():
                    start_at = time.monotonic()
                    if not self.following:
                        tracker.reset()
                        self.stop_event.wait(0.07)
                        continue
                    with self.flow_lock:
                        generation = self.flow_generation
                        seed_box = self.flow_seed_box
                    shot = np.asarray(screen.grab(monitor))
                    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                    frame = resize_for_model(frame)
                    if previous_generation != generation:
                        previous_generation = generation
                        tracker.reset()
                        if seed_box is not None:
                            tracker.seed(frame, seed_box)
                    elif seed_box is not None:
                        box = tracker.update(frame)
                        if box is not None:
                            payload = (generation, box, frame.shape)
                            try:
                                while True:
                                    self.flow_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.flow_queue.put_nowait(payload)
                            except queue.Full:
                                pass
                    self.stop_event.wait(max(.004, .033 - (time.monotonic() - start_at)))
        except Exception as exc:
            self._post("message", f"Fluxo óptico indisponível ({exc}). Continuando com YOLO.")

    def _acompanhar_quadro(self, frame, detections):
        """Usa YOLO para detectar e fluxo óptico para acompanhar entre inferências."""
        if not self.following:
            self.active_box = None
            return
        filtrados = self._filtered_detections(detections, frame.shape)
        agora = time.monotonic()
        fast_recente = (self.source.get() == "tela" and
                        self.active_box is not None and
                        agora - self.fast_last_at < .35)
        if fast_recente:
            # Confirma a posição de fluxo por uma nova detecção de pessoa.
            x1, y1, x2, y2 = self.active_box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            largura, altura = x2 - x1, y2 - y1
            candidates = []
            for det in filtrados:
                a, b, d, e = det["box"]
                tx, ty = (a + d) / 2, (b + e) / 2
                distancia = ((cx - tx) ** 2 + (cy - ty) ** 2) ** .5
                if distancia < max(45, largura * 1.4, altura * .65):
                    candidates.append((distancia, det))
            if candidates:
                melhor = min(candidates, key=lambda item: item[0])[1]
                self.selected_id = melhor["id"]
                self.yolo_last_confirmed = agora
                # Corrige desvio acumulado do fluxo, mas não reinicia a cada quadro.
                if agora - self.last_flow_seed_at > 1.2:
                    self._set_flow_seed(melhor["box"])
                return  # O fluxo rápido continua guiando a posição entre inferências.
            elif agora - self.yolo_last_confirmed < 1.1:
                return
            else:
                self._set_flow_seed(None)
                self.active_box = None
                self.last_target_box = None
                self.selected_id = None

        escolhido = self._selecionar_automaticamente(filtrados, frame.shape)
        if escolhido is None:
            return
        self.active_box = escolhido["box"]
        self.yolo_last_confirmed = agora
        self.last_target_box = escolhido["box"]
        if self.source.get() == "tela":
            self._set_flow_seed(self.active_box)
        # Enquanto o fluxo inicializa, o YOLO também pode corrigir o mouse.
        self._move_camera_for_box(self.active_box, frame.shape)

    def _move_camera_for_box(self, box, frame_shape):
        if not self.following or not self.enable_mouse.get() or self.source.get() != "tela":
            return
        if self.only_fortnite.get() and not janela_fortnite_ativa():
            now = time.monotonic()
            if now - self.last_control_notice_at > 2.0:
                self.last_control_notice_at = now
                self.info.set("Movimento pausado: Fortnite não está em primeiro plano.")
            return
        # Evita enviar entradas em alta frequência, que causariam trepidação.
        now = time.monotonic()
        if now-self.last_move_at < .020:
            return
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        # Mira aproximadamente no centro visual do corpo, não em um ponto hostil.
        target_x = (x1+x2)/2
        target_y = y1+(y2-y1)*.46
        # Correção rápida quando longe; mais suave quando quase centralizado.
        erro_maior = max(abs(target_x - w / 2), abs(target_y - h / 2))
        alpha = 0.87 if erro_maior > 110 else (0.70 if erro_maior > 35 else 0.48)
        if self.smooth_target is None:
            self.smooth_target = (target_x, target_y)
        else:
            sx, sy = self.smooth_target
            self.smooth_target = (sx * (1 - alpha) + target_x * alpha,
                                  sy * (1 - alpha) + target_y * alpha)
        target_x, target_y = self.smooth_target
        dx, dy = calcular_passo(target_x-w/2, target_y-h/2,
                               ganho=self.mouse_gain.get(), limite=60, zona_morta=6)
        if dx == 0 and dy == 0:
            return
        ok = mover_relativo(dx, dy)
        self.last_move_at = now
        if ok and now - self.last_control_notice_at > 3.0:
            self.last_control_notice_at = now
            self.info.set(
                "Windows recebeu movimento do mouse; o Fortnite pode ignorar. "
                "ESC interrompe."
            )
        if not ok and not self.mouse_failure_warned:
            self.mouse_failure_warned = True
            self.following = False
            self.info.set("O Windows não aceitou o movimento de mouse; controle pausado.")

    def _draw(self, frame, detections):
        import cv2
        from PIL import Image, ImageTk
        annotated = frame.copy()
        detections = self._filtered_detections(detections, frame.shape)
        h, w = annotated.shape[:2]
        selected_box = self.active_box if self.following else None
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            selected = selected_box is not None and det["box"] == selected_box
            color = (10, 212, 255) if selected else (120, 230, 100)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if selected else 2)
            tag = str(det["id"]) if det["id"] is not None else "sem ID"
            cv2.putText(annotated, f"ID {tag} - {det['confidence']:.2f}",
                        (max(0, x1), max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, color, 2, cv2.LINE_AA)
        center = (w // 2, h // 2)
        cv2.drawMarker(annotated, center, (225, 225, 225), markerType=cv2.MARKER_CROSS,
                       markerSize=22, thickness=2)
        if self.following and selected_box is not None:
            x1, y1, x2, y2 = selected_box
            aim = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.arrowedLine(annotated, center, aim, (10, 212, 255), 2, tipLength=.15)
            # O movimento é tratado antes da renderização, em _acompanhar_quadro.
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        dx, dy = (PREVIEW_W - dw) // 2, (PREVIEW_H - dh) // 2
        rgb = cv2.cvtColor(cv2.resize(annotated, (dw, dh)), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(dx, dy, image=photo, anchor="nw")
        self.last_photo = photo
        self.preview_mapping = (scale, dx, dy)

    def _poll(self):
        try:
            while True:
                key = self.key_commands.get_nowait()
                if key == "p":
                    self.toggle_follow()
                elif key == "esc":
                    self.release_target()
        except queue.Empty:
            pass

        # Dá prioridade às coordenadas recentes do fluxo óptico.
        try:
            while True:
                generation, box, frame_shape = self.flow_queue.get_nowait()
                with self.flow_lock:
                    valido = generation == self.flow_generation
                if valido and self.following and self.source.get() == "tela":
                    self.active_box = box
                    self.last_target_box = box
                    self.last_target_at = time.monotonic()
                    self.fast_last_at = time.monotonic()
                    self.fast_frames_seen += 1
                    self._move_camera_for_box(box, frame_shape)
        except queue.Empty:
            pass

        # O detector YOLO usa sua própria fila, sem bloquear o rastreador rápido.
        try:
            payload = self.frame_queue.get_nowait()
            self.latest_frame, self.latest_boxes = payload
            self.yolo_frames_seen += 1
            self._acompanhar_quadro(*payload)
            now = time.monotonic()
            if now - self.last_draw_at >= .125:
                self._draw(*payload)
                self.last_draw_at = now
        except queue.Empty:
            pass

        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "message":
                    self.info.set(payload)
                elif kind == "error":
                    self.info.set(payload)
                    messagebox.showerror("HUD V8", payload)
                elif kind == "done":
                    self.stop_event.set()
                    self.running = False
                    self.following = False
                    self.active_box = None
        except queue.Empty:
            pass

        now = time.monotonic()
        elapsed = now - self.metrics_started_at
        if elapsed >= 1.5:
            self.metrics.set(
                f"YOLO: {self.yolo_frames_seen/elapsed:.1f} FPS | "
                f"Fluxo: {self.fast_frames_seen/elapsed:.1f} FPS"
            )
            self.yolo_frames_seen = 0
            self.fast_frames_seen = 0
            self.metrics_started_at = now
        self.root.after(20, self._poll)

    def close(self):
        self.stop_event.set()
        if self.listener is not None:
            self.listener.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 11):
        raise SystemExit("É necessário Python 3.11 ou 3.12 para este instalador.")
    HUD().run()
