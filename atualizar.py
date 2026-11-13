"""Atualizador manual do HUD. Baixa apenas arquivos conhecidos do próprio GitHub.

Não modifica .venv, modelos .pt nem preferências locais.
Feche o HUD antes de executar.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OWNER = "kurofunways-cpu"
REPO = "HUD-Fortnite"
BRANCH = "main"
BASE = Path(__file__).resolve().parent
ARQUIVOS = (
    "app.py", "movimento.py", "iniciar.bat", "instalar.bat",
    "requirements.txt", "LEIA_PRIMEIRO.txt",
)
LIMITE = 2_000_000


def baixar(url: str) -> bytes:
    req = Request(url, headers={
        "User-Agent": "HUD-Fortnite-Atualizador/1.0",
        "Accept": "application/vnd.github+json" if "api.github.com" in url else "*/*",
    })
    with urlopen(req, timeout=20) as response:
        conteudo = response.read(LIMITE + 1)
    if len(conteudo) > LIMITE:
        raise ValueError("Arquivo remoto grande demais. Atualização cancelada.")
    return conteudo


def sha_git(dados: bytes) -> str:
    # GitHub usa o hash do objeto Git: "blob <tamanho>\\0<conteúdo>".
    cabecalho = f"blob {len(dados)}".encode("ascii") + bytes([0])
    return hashlib.sha1(cabecalho + dados).hexdigest()


def executar() -> int:
    url_api = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
    arvore = json.loads(baixar(url_api).decode("utf-8"))
    if arvore.get("truncated"):
        print("Lista do repositório incompleta; atualização cancelada.")
        return 1
    remotos = {
        item["path"]: item["sha"]
        for item in arvore.get("tree", [])
        if item.get("type") == "blob" and item.get("path") in ARQUIVOS
    }
    if not remotos:
        print("Nenhuma versão do programa foi publicada no repositório ainda.")
        return 0

    pendentes = {}
    for nome in ARQUIVOS:
        if nome not in remotos:
            continue
        local = BASE / nome
        if local.exists() and sha_git(local.read_bytes()) == remotos[nome]:
            continue
        print("Baixando:", nome)
        url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{nome}"
        dados = baixar(url)
        if sha_git(dados) != remotos[nome]:
            raise ValueError(f"Verificação de integridade falhou: {nome}. Tente novamente.")
        pendentes[nome] = dados

    if not pendentes:
        print("Tudo atualizado. Nenhum download necessário.")
        return 0

    backup_dir = BASE / ".backup_atualizacao"
    backup_dir.mkdir(exist_ok=True)
    # Prepara todos os downloads antes de substituir qualquer arquivo.
    with tempfile.TemporaryDirectory(prefix="hud-update-") as temporaria:
        tmp = Path(temporaria)
        for nome, dados in pendentes.items():
            (tmp / nome).write_bytes(dados)
        for nome in pendentes:
            arquivo = BASE / nome
            if arquivo.is_file():
                shutil.copy2(arquivo, backup_dir / nome)
        for nome in pendentes:
            os.replace(tmp / nome, BASE / nome)
            print("Atualizado:", nome)

    print("Atualização concluída; seu ambiente .venv foi preservado.")
    if "requirements.txt" in pendentes:
        print("As dependências mudaram. Execute instalar.bat antes de iniciar.")
    return 0


if __name__ == "__main__":
    print("HUD Fortnite - Atualizador manual")
    print("Feche o HUD antes de atualizar. Fonte:", f"github.com/{OWNER}/{REPO}")
    try:
        sys.exit(executar())
    except (OSError, URLError, HTTPError, ValueError, KeyError, json.JSONDecodeError) as erro:
        print("Falha na atualização:", erro)
        sys.exit(1)
