"""Client de test en ligne de commande : envoie un wav lingala, récupère la voix.

    python test_client.py mon_audio.wav
    python test_client.py mon_audio.wav wss://simba-tts.chronoconcept.net

Le wav doit être mono 16 kHz 16 bits :
    ffmpeg -i enregistrement.m4a -ac 1 -ar 16000 -sample_fmt s16 mon_audio.wav

Dépend de « websockets », déjà installé par uvicorn[standard].
"""

import asyncio
import json
import sys
import wave

import websockets

SR = 16000
FRAME = 480 * 2  # 30 ms de PCM 16 bits, comme le navigateur et la VAD


async def main(path: str, base: str) -> None:
    with wave.open(path, "rb") as w:
        if (w.getnchannels(), w.getframerate(), w.getsampwidth()) != (1, SR, 2):
            sys.exit(f"{path} doit être mono, {SR} Hz, 16 bits (voir l'entête du fichier).")
        pcm = w.readframes(w.getnframes())
    print(f"{path} : {len(pcm) / 2 / SR:.1f} s d'audio")

    url = base.rstrip("/") + "/ws/conversation"
    reply = bytearray()
    async with websockets.connect(url, max_size=None) as ws:

        async def pump() -> None:
            # On respecte le temps réel : la VAD compte les trames de silence,
            # un envoi en rafale lui ferait croire que le tour est fini d'emblée.
            for i in range(0, len(pcm), FRAME):
                await ws.send(pcm[i : i + FRAME])
                await asyncio.sleep(0.03)
            for _ in range(33):  # ~1 s de silence pour clore le tour
                await ws.send(b"\x00" * FRAME)
                await asyncio.sleep(0.03)

        task = asyncio.create_task(pump())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    reply += msg
                    print(f"  audio +{len(msg) / 2 / SR:.1f} s")
                    continue
                event = json.loads(msg)
                print(f"[{event['type']}] {event.get('text') or event.get('message') or ''}")
                if event["type"] == "turn_end":
                    break
        finally:
            task.cancel()

    if not reply:
        sys.exit("aucun audio reçu — voir les logs du serveur.")
    with wave.open("reply.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(reply)
    print(f"-> reply.wav ({len(reply) / 2 / SR:.1f} s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000"))
