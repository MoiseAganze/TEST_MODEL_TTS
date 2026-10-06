"""API de conversation vocale en lingala.

Un seul endpoint utile : /ws/conversation. Le client envoie du PCM 16 kHz mono
16 bits petit-boutiste en trames binaires, le serveur répond par des événements
JSON (texte) et par de l'audio au même format (trames binaires).

Protocole, côté client :
  binaire        -> chunk de micro (n'importe quelle taille)
  {"type":"reset"} -> oublie l'historique de la conversation

Côté serveur :
  {"type":"listening"}             le tour de parole est détecté, ça travaille
  {"type":"transcript","text":...} ce qui a été entendu
  {"type":"reply","text":...}      une phrase de la réponse (juste avant son audio)
  binaire                          le PCM de cette phrase
  {"type":"turn_end","ms":...}     fin du tour, latence totale
  {"type":"error","message":...}
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from time import perf_counter

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import llm
import stt
import tts
from config import cpu_slot, settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-7s %(name)-6s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ws")

# Assez pour garder le fil d'une conversation, assez peu pour ne pas gonfler le
# prompt (donc la latence) à chaque tour.
HISTORY_MAX = 12


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Les modèles sont chargés UNE fois, ici. torch doit être bridé avant la
    # première inférence : sinon il prend les 8 vCPU et deux conversations
    # simultanées se marchent dessus.
    torch.set_num_threads(settings.torch_threads)
    log.info("chargement des modèles (%d threads torch)…", settings.torch_threads)
    t0 = perf_counter()
    await asyncio.gather(asyncio.to_thread(stt.load), asyncio.to_thread(tts.load))
    llm.load()
    log.info("prêt en %.1f s — %d conversations simultanées max",
             perf_counter() - t0, settings.max_concurrent)
    yield
    await llm.close()


app = FastAPI(title="Simba Lingala Voice", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr": settings.asr_model,
        "tts": settings.tts_model,
        "llm": settings.llm_model,
        # _value : pas d'API publique pour ça, et savoir si on sature les vCPU
        # en prod vaut bien un attribut privé.
        "slots_libres": cpu_slot._value,
    }


async def handle_turn(ws: WebSocket, pcm, history: list[dict]) -> None:
    """Un tour complet : transcription -> LLM -> voix, phrase par phrase."""
    t0 = perf_counter()
    try:
        await ws.send_json({"type": "listening"})
        log.info("audio reçu : %.1f s", len(pcm) / settings.sample_rate)

        text = await stt.transcribe(pcm)
        log.info("transcription (%.1f s) : %r", perf_counter() - t0, text)
        if not text:
            log.info("rien de compris, tour abandonné")
            return
        await ws.send_json({"type": "transcript", "text": text})

        history.append({"role": "user", "content": text})
        spoken: list[str] = []
        async for sentence in llm.stream_reply(history):
            log.info("réponse (+%.1f s) : %r", perf_counter() - t0, sentence)
            await ws.send_json({"type": "reply", "text": sentence})
            if (audio := await tts.synthesize(sentence)) is not None:
                await ws.send_bytes(audio.tobytes())
                log.info("voix envoyée : %.1f s d'audio (+%.1f s)",
                         len(audio) / settings.sample_rate, perf_counter() - t0)
            spoken.append(sentence)

        if spoken:
            history.append({"role": "assistant", "content": " ".join(spoken)})
        del history[:-HISTORY_MAX]
    except Exception as exc:
        # Rien ne doit remonter de cette tâche : une exception non récupérée
        # ferait juste un warning asyncio illisible dans les logs de prod.
        if isinstance(exc, WebSocketDisconnect):
            log.info("client parti pendant le tour")
            return
        log.exception("tour en échec")
        with suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        with suppress(Exception):
            await ws.send_json({"type": "turn_end", "ms": int((perf_counter() - t0) * 1000)})


@app.websocket("/ws/conversation")
async def conversation(ws: WebSocket) -> None:
    await ws.accept()
    log.info("connexion ouverte")
    detector = stt.TurnDetector()
    history: list[dict] = []
    turn: asyncio.Task | None = None
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            if turn is not None:
                if not turn.done():
                    # Anti-écho : pendant que l'assistant parle, le micro du
                    # client capte sa propre voix. On jette cet audio au lieu de
                    # le transcrire.
                    continue
                turn = None
                detector.reset()

            if (chunk := message.get("bytes")) is not None:
                if (utterance := detector.feed(chunk)) is not None:
                    turn = asyncio.create_task(handle_turn(ws, utterance, history))
            elif message.get("text"):
                # Un seul message de contrôle, inutile de sortir un schéma pour ça.
                if '"reset"' in message["text"]:
                    history.clear()
                    detector.reset()
                    log.info("historique remis à zéro")
    except WebSocketDisconnect:
        pass
    finally:
        if turn is not None and not turn.done():
            turn.cancel()
        log.info("connexion fermée")


# Monté en dernier : une route montée sur « / » avale tout ce qui passe après
# elle. static/index.html est la page de test au micro.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
