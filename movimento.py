"""Controlador relativo de mouse para Windows, sem injeção/leitura de memória.

Retorna False quando não há janela do Fortnite ativa, em outro sistema
ou caso o Windows rejeite o evento de entrada. Não contorna anticheat.
"""
from __future__ import annotations
import ctypes
import sys
from ctypes import wintypes


def calcular_passo(erro_x: float, erro_y: float, ganho: float = 0.14,
                   limite: int = 38, zona_morta: float = 13.0) -> tuple[int, int]:
    def eixo(valor):
        if abs(valor) <= zona_morta:
            return 0
        passo = max(-limite, min(limite, valor * ganho))
        return int(round(passo)) or (1 if valor > 0 else -1)
    return eixo(erro_x), eixo(erro_y)


def nome_janela_ativa() -> str:
    if sys.platform != 'win32':
        return ''
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return ''
    buf = ctypes.create_unicode_buffer(512)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


def janela_fortnite_ativa() -> bool:
    return 'fortnite' in nome_janela_ativa().lower()


def mover_relativo(dx: int, dy: int) -> bool:
    if sys.platform != 'win32':
        return False
    if dx == 0 and dy == 0:
        return True
    # Estrutura INPUT para SendInput, ABI Windows x86 / x64.
    ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG),
                    ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
                    ('time', wintypes.DWORD), ('dwExtraInfo', ULONG_PTR)]
    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD),
                    ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
                    ('dwExtraInfo', ULONG_PTR)]
    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [('uMsg', wintypes.DWORD), ('wParamL', wintypes.WORD),
                    ('wParamH', wintypes.WORD)]
    class INPUTUNION(ctypes.Union):
        _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]
    class INPUT(ctypes.Structure):
        _fields_ = [('type', wintypes.DWORD), ('data', INPUTUNION)]
    event = INPUT(type=0, data=INPUTUNION(mi=MOUSEINPUT(dx=dx, dy=dy,
                         mouseData=0, dwFlags=0x0001, time=0, dwExtraInfo=0)))
    send_input = ctypes.windll.user32.SendInput
    send_input.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    send_input.restype = wintypes.UINT
    return send_input(1, ctypes.byref(event), ctypes.sizeof(INPUT)) == 1
