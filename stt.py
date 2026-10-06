"""STT lingala : VAD (découpage des tours de parole) + transcription CTC.

La VAD vit ici et pas dans son propre module : c'est le même sujet, le flux audio
entrant, et ça fait trente lignes.
"""

import asyncio
import logging

import numpy as np
import torch
import webrtcvad
from transformers import AutoModelForCTC, AutoProcessor

from config import cpu_slot, settings

log = logging.getLogger(__name__)

_model = None
_processor = None


class TurnDetector:
    """Transforme un flux PCM continu en tours de parole.

    webrtcvad n'accepte que des trames de 10, 20 ou 30 ms : on tamponne les
    octets reçus et on les consomme par trames de 30 ms.
    """

    FRAME_MS = 30
    PREROLL = 5  # ~150 ms gardés avant la parole, sinon la 1re syllabe est coupée

    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self._frame_bytes = settings.sample_rate * self.FRAME_MS // 1000 * 2
        self._silence_frames = settings.silence_ms // self.FRAME_MS
        self._min_speech_frames = settings.min_speech_ms // self.FRAME_MS
        self._max_frames = settings.max_turn_s * 1000 // self.FRAME_MS
        self.reset()

    def reset(self) -> None:
        self._buf = b""
        self._frames: list[bytes] = []
        self._speech = 0
        self._quiet = 0
        self._started = False

    def feed(self, pcm: bytes) -> np.ndarray | None:
        """Renvoie le tour de parole complet dès qu'il est terminé, sinon None."""
        self._buf += pcm
        while len(self._buf) >= self._frame_bytes:
            frame = self._buf[: self._frame_bytes]
            self._buf = self._buf[self._frame_bytes :]
            if (turn := self._push(frame)) is not None:
                return turn  # le reste du tampon attendra le prochain appel
        return None

    def _push(self, frame: bytes) -> np.ndarray | None:
        voiced = self._vad.is_speech(frame, settings.sample_rate)
        self._frames.append(frame)

        if not self._started:
            if voiced:
                self._speech += 1
            else:
                self._speech = 0
                # On ne garde qu'un court pré-roll : tout l'historique, ce serait
                # des minutes de silence en mémoire par connexion.
                del self._frames[: -self.PREROLL]
            if self._speech < self._min_speech_frames:
                return None
            self._started = True
            self._quiet = 0
            return None

        if voiced:
            self._quiet = 0
        else:
            self._quiet += 1
            if self._quiet >= self._silence_frames:
                return self._emit()
        if len(self._frames) >= self._max_frames:
            log.info("tour coupé à %d s (max_turn_s)", settings.max_turn_s)
            return self._emit()
        return None

    def _emit(self) -> np.ndarray:
        pcm = np.frombuffer(b"".join(self._frames), dtype=np.int16)
        self.reset()
        return pcm


def load() -> None:
    global _model, _processor
    kwargs = {}
    if settings.asr_adapter:
        kwargs = {"target_lang": settings.asr_adapter, "ignore_mismatched_sizes": True}
    _processor = AutoProcessor.from_pretrained(settings.asr_model)
    _model = AutoModelForCTC.from_pretrained(settings.asr_model, **kwargs).eval()
    log.info("STT chargé : %s%s", settings.asr_model,
             f" (adaptateur {settings.asr_adapter})" if settings.asr_adapter else "")


def _run(pcm: np.ndarray) -> str:
    audio = pcm.astype(np.float32) / 32768.0
    inputs = _processor(audio, sampling_rate=settings.sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
    # CTC : un seul passage, argmax, pas de décodage autoregressif. C'est toute
    # la raison pour laquelle ça tient le temps réel sur CPU.
    return _processor.batch_decode(logits.argmax(dim=-1))[0].strip()


async def transcribe(pcm: np.ndarray) -> str:
    async with cpu_slot:
        return await asyncio.to_thread(_run, pcm)


if __name__ == "__main__":
    # Self-check de la VAD : c'est la logique la plus facile à casser (compteurs,
    # pré-roll, seuils). Aucun modèle requis.
    d = TurnDetector()
    frame = d._frame_bytes
    silence = b"\x00" * frame
    # du bruit fort : webrtcvad le prend pour de la parole
    rng = np.random.default_rng(0)
    speech = (rng.normal(0, 6000, frame // 2)).astype(np.int16).tobytes()

    assert d.feed(silence * 20) is None, "du silence ne doit pas ouvrir un tour"
    assert d.feed(speech * 20) is None, "un tour ne se ferme pas pendant la parole"
    turn = d.feed(silence * 40)
    assert turn is not None, "700 ms de silence doivent clore le tour"
    assert turn.dtype == np.int16 and len(turn) > frame // 2 * 20
    assert not d._started and d._frames == [], "reset() après émission"
    # un hoquet de 300 ms au milieu d'une phrase ne doit pas la couper en deux
    d.feed(speech * 20)
    assert d.feed(silence * 10) is None
    assert d.feed(speech * 10) is None
    print("stt.py: VAD OK")
