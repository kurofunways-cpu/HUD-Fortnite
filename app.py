"""HUD Aliado V4 — rastreamento visual e controle EXPERIMENTAL de mouse.

Não lê memória, não injeta código e não contorna anticheat. O Fortnite pode
ignorar os eventos de mouse. Use por sua conta e risco; pode violar as regras.
"""

from __future__ import annotations

import queue
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


class HUD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HUD V6 | Alvo automático")
        self.root.geometry("1030x790+90+70")
        self.root.minsize(1000, 760)
        self.root.configure(bg=DARK)
        self.messages = queue.Queue(maxsize=3)
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
        self.mouse_gain = tk.DoubleVar(value=0.12)
        self.smooth_target = None
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
        title = self._label(header, "  HUD V6  •  seleção automática + YOLO + BoT-SORT", ("Segoe UI", 13, "bold"), bg="#253650")
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
        self._button(row, "P: Travar / soltar", self.toggle_follow).pack(side="left", padx=8)

        self.info = tk.StringVar(value="Pronto. Escolha uma fonte e clique em Iniciar prévia.")
        self._label(self.root, "P seleciona automaticamente personagem próximo da mira; ESC solta.",
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
        tk.Scale(speed, from_=0.03, to=0.48, resolution=0.01, orient="horizontal",
                 length=270, variable=self.mouse_gain, bg=DARK, fg=TEXT,
                 highlightthickness=0, troughcolor=PANEL).pack(side="left")
        self._button(speed, "Testar mouse no Windows", self.test_mouse).pack(side="left", padx=8)
        self._label(speed, "ESC solta; P liga/desliga.",
                    ("Segoe UI", 9), fg="#f8c58c").pack(side="left", padx=10)

        self.canvas = tk.Canvas(self.root, width=PREVIEW_W, height=PREVIEW_H,
                                highlightthickness=1, highlightbackground="#466079", bg="#090e16")
        self.canvas.pack(padx=12)
        # Sem seleção manual necessária
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                                text="A prévia aparece aqui depois de iniciar", fill="#87a0b7",
                                font=("Segoe UI", 16))
        self._label(self.root, "Status", ("Segoe UI", 10, "bold"), fg=ACCENT).pack(anchor="w", padx=16, pady=(10, 2))
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
            try:
                while True:
                    self.messages.get_nowait()
            except queue.Empty:
                pass
        try:
            self.messages.put_nowait((kind, payload))
        except queue.Full:
            # Evita bloquear a thread se a interface estiver ocupada.
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
        self.following = False
        self.selected_id = None
        self.smooth_target = None
        self.latest_boxes = []
        self.running = True
        src, path = self.source.get(), self.video_path
        self.thread = threading.Thread(target=self._worker, args=(src, path), daemon=True)
        self.thread.start()
        self.info.set("Preparando modelo de IA. O primeiro uso precisa de internet...")

    def _worker(self, source, path):
        capture = None
        try:
            import cv2
            import mss
            import numpy as np
            from ultralytics import YOLO
            model = YOLO("yolo11n.pt")
            self._post("message", "Modelo pronto. Detectando pessoas na imagem...")
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
                                          classes=[0], conf=0.30, imgsz=640, verbose=False,
                                          device="cpu")
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
        self.following = False
        self.selected_id = None
        self.info.set("Parando prévia e movimento de mouse...")

    def toggle_follow(self):
        if self.following:
            self.release_target()
            return
        if not self.running:
            self.info.set("Inicie a prévia antes de apertar P.")
            return
        self.selected_id = None
        self.smooth_target = None
        self.following = True
        self.lost_frames = 0
        self.mouse_failure_warned = False
        if self.enable_mouse.get() and self.source.get() == "tela":
            self.info.set("Alvo automático ativo; ESC interrompe.")
        else:
            self.info.set("Alvo automático: prévia visual, sem controle do mouse.")

    def release_target(self):
        self.following = False
        self.selected_id = None
        self.smooth_target = None
        self.info.set("ESC: acompanhamento interrompido. O mouse não será mais movido.")

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
        """Mantém um ID; procura o mais próximo do centro quando necessário."""
        h, w = frame_shape[:2]
        validos = [d for d in detections if d["id"] is not None]
        if self.selected_id is not None:
            atual = next((d for d in validos if d["id"] == self.selected_id), None)
            if atual is not None:
                self.lost_frames = 0
                return atual
            self.lost_frames += 1
            if self.lost_frames <= 5:
                return None
            self.selected_id = None
            self.smooth_target = None
        self.lost_frames = 0
        if not validos:
            return None
        cx, cy = w / 2, h / 2
        limite = 0.83 * ((cx * cx + cy * cy) ** 0.5)
        candidatos = []
        for d in validos:
            x1, y1, x2, y2 = d["box"]
            alvo_x, alvo_y = (x1 + x2) / 2, y1 + (y2 - y1) * 0.46
            distancia = ((alvo_x - cx) ** 2 + (alvo_y - cy) ** 2) ** 0.5
            if distancia <= limite:
                candidatos.append((distancia, -d["confidence"], d))
        if not candidatos:
            return None
        melhor = min(candidatos, key=lambda item: item[:2])[2]
        self.selected_id = melhor["id"]
        return melhor

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
        if now-self.last_move_at < .065:
            return
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        # Mira aproximadamente no centro visual do corpo, não em um ponto hostil.
        target_x = (x1+x2)/2
        target_y = y1+(y2-y1)*.46
        if self.smooth_target is None:
            self.smooth_target = (target_x, target_y)
        else:
            sx, sy = self.smooth_target
            self.smooth_target = (sx * 0.70 + target_x * 0.30,
                                  sy * 0.70 + target_y * 0.30)
        target_x, target_y = self.smooth_target
        dx, dy = calcular_passo(target_x-w/2, target_y-h/2,
                               ganho=self.mouse_gain.get(), limite=24, zona_morta=10)
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
        escolhido = self._selecionar_automaticamente(detections, frame.shape) if self.following else None
        selected_box = escolhido["box"] if escolhido is not None else None
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            selected = self.selected_id is not None and det["id"] == self.selected_id
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
            self.lost_frames = 0
            self._move_camera_for_box(selected_box, frame.shape)
        elif self.following:
            # Aguarda novas detecções para adquirir outro alvo.
            pass
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
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "frame":
                    self.latest_frame, self.latest_boxes = payload
                    self._draw(*payload)
                elif kind == "message":
                    self.info.set(payload)
                elif kind == "error":
                    self.info.set(payload)
                    messagebox.showerror("HUD Aliado V4", payload)
                elif kind == "done":
                    self.running = False
                    self.following = False
        except queue.Empty:
            pass
        self.root.after(70, self._poll)

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
