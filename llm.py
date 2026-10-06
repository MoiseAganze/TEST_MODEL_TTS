"""Le « cerveau » : GPT-4o via OpenRouter, en streaming, découpé en phrases.

UN SEUL appel, pas trois (lingala->français, réponse, français->lingala). Trois
allers-retours ajoutent 3 à 6 s de latence et surtout interdisent le streaming :
impossible de commencer le TTS avant la fin de la 3e requête. Le system prompt
obtient le même bénéfice — GPT-4o raisonne en français, langue où il est bien
meilleur qu'en lingala — en un seul aller-retour, et les phrases lingala
arrivent au fil de l'eau : la première part au TTS pendant que la suite
se génère.
"""

import json
import logging
import re
from collections.abc import AsyncIterator

import httpx

from config import settings

log = logging.getLogger(__name__)

URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM = """Tu es un assistant vocal qui converse en lingala (RDC).

Méthode : raisonne en français dans ta tête, puis réponds UNIQUEMENT en lingala.
N'écris jamais de français dans ta réponse.

Ta réponse part directement dans un synthétiseur vocal, donc :
- 1 à 3 phrases courtes, chacune terminée par un point.
- Jamais de liste, de markdown, de guillemets, de parenthèses ni d'emoji.
- Les nombres s'écrivent en mots lingala (« mibale », jamais « 2 »).
- Jamais de réponse d'un seul mot : toujours une phrase complète.

Le texte que tu reçois sort d'un modèle de reconnaissance vocale et contient des
erreurs. Devine l'intention la plus plausible au lieu de demander une
répétition."""

# Frontière de phrase : on coupe après la ponctuation finale, dès que le token
# suivant est arrivé (c'est l'espace qui confirme la fin).
_SPLIT = re.compile(r"(?<=[.!?…])\s")
# En dessous, le TTS écarterait la phrase (MIN_LEN) : on la fusionne avec la
# suivante plutôt que de la perdre.
MIN_CHUNK = 12
# Au-delà, on coupe sans attendre de ponctuation : sinon un LLM bavard fait
# patienter le TTS jusqu'à la fin de sa génération.
MAX_CHUNK = 180

_client: httpx.AsyncClient | None = None


def load() -> None:
    global _client
    if not settings.openrouter_api_key:
        log.warning("OPENROUTER_API_KEY est vide : le LLM répondra par une erreur.")
    # Client réutilisé : une poignée de main TLS par tour de parole, ce sont
    # 100 à 200 ms de latence offerts à chaque fois.
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=10.0),
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "X-Title": "Simba Lingala Voice",
        },
    )


async def close() -> None:
    if _client is not None:
        await _client.aclose()


def _cut(buf: str) -> tuple[str, str] | None:
    """Renvoie (phrase, reste) dès qu'une phrase exploitable est disponible."""
    for m in _SPLIT.finditer(buf):
        if m.start() >= MIN_CHUNK:
            return buf[: m.start()].strip(), buf[m.start() :].lstrip()
    if len(buf) > MAX_CHUNK:
        if (i := buf.rfind(" ", MIN_CHUNK, MAX_CHUNK)) > 0:
            return buf[:i].strip(), buf[i:].lstrip()
    return None


async def stream_reply(history: list[dict]) -> AsyncIterator[str]:
    """Donne les phrases lingala une par une, au fur et à mesure de la génération."""
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "system", "content": SYSTEM}, *history],
        "stream": True,
        "temperature": 0.6,
        "max_tokens": 300,
    }
    buf = ""
    async with _client.stream("POST", URL, json=payload) as resp:
        if resp.status_code >= 400:
            detail = (await resp.aread()).decode(errors="replace")[:300]
            raise RuntimeError(f"OpenRouter {resp.status_code} : {detail}")
        async for line in resp.aiter_lines():
            # OpenRouter intercale des lignes de commentaire SSE (« : OPENROUTER
            # PROCESSING ») pendant l'attente : le préfixe les écarte.
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                piece = json.loads(data)["choices"][0]["delta"].get("content") or ""
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            buf += piece
            while cut := _cut(buf):
                sentence, buf = cut
                yield sentence
    if rest := buf.strip():
        yield rest


if __name__ == "__main__":
    # Self-check du découpage en phrases : c'est lui qui garantit au TTS des
    # phrases ni vides, ni trop courtes, ni interminables. Aucun appel réseau.
    def split_all(text: str, step: int = 7) -> list[str]:
        """Simule l'arrivée du texte par petits bouts, comme en streaming."""
        out, buf = [], ""
        for i in range(0, len(text), step):
            buf += text[i : i + step]
            while cut := _cut(buf):
                sentence, buf = cut
                out.append(sentence)
        if buf.strip():
            out.append(buf.strip())
        return out

    assert split_all("Mbote na yo. Nazali malamu. Boni yo?") == [
        "Mbote na yo.", "Nazali malamu.", "Boni yo?"
    ]
    # une phrase trop courte est fusionnée avec la suivante, pas perdue
    assert split_all("Ee. Nazali kondima na yo mingi.") == ["Ee. Nazali kondima na yo mingi."]
    # pas de ponctuation du tout : coupe de secours sur un espace
    long = "lokasa " * 40
    chunks = split_all(long)
    assert all(len(c) <= MAX_CHUNK for c in chunks), [len(c) for c in chunks]
    assert "".join(chunks).replace(" ", "") == long.replace(" ", "")
    assert split_all("") == [] and split_all("   ") == []
    assert all(c.strip() for c in split_all("A. B. C. Toza wapi na yo lelo mpenza?"))
    print("llm.py: découpage OK")
