import argparse
import asyncio
import base64
import json
import logging
import re
import signal
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1p1beta1 as speech

app = FastAPI()

# Audio recording parameters
RATE = 8000
CHUNK = int(RATE / 10)  # 100ms


def signal_handler(sig, frame):
    sys.exit(0)


def listen_print_loop(responses):
    """Iterates through recognition responses and prints them."""
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break
            num_chars_printed = 0


class AsyncStream:
    """
    An async version of the audio stream to use with asyncio.
    (For demonstration, we only log media messages.)
    """

    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self.buff = asyncio.Queue()
        self.closed = True

    async def __aenter__(self):
        self.closed = False
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        await self.buff.put(None)

    async def fill_buffer(self, in_data):
        await self.buff.put(in_data)

    async def generator(self):
        while True:
            chunk = await self.buff.get()
            if chunk is None:
                return
            data = [chunk]
            # Drain the queue
            while not self.buff.empty():
                chunk = self.buff.get_nowait()
                if chunk is None:
                    return
                data.append(chunk)
            yield b"".join(data)


@app.websocket("/media")
async def media_socket(websocket: WebSocket):
    await websocket.accept()
    logging.getLogger("uvicorn.error").info("WebSocket connection accepted")
    while True:
        try:
            message = await websocket.receive_text()
        except WebSocketDisconnect:
            logging.getLogger("uvicorn.error").info("WebSocket disconnected")
            break

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logging.getLogger("uvicorn.error").error(
                "Invalid JSON received: {}".format(message)
            )
            continue

        event = data.get("event")
        if event == "connected":
            logging.getLogger("uvicorn.error").info(
                "Connected Message received: {}".format(message)
            )
        elif event == "start":
            logging.getLogger("uvicorn.error").info(
                "Start Message received: {}".format(message)
            )
        elif event == "media":
            payload = data.get("media", {}).get("payload")
            if payload:
                try:
                    print("Received media message: {}".format(payload))
                    chunk = base64.b64decode(payload)
                except Exception as e:
                    logging.getLogger("uvicorn.error").error(
                        "Error decoding payload: {}".format(e)
                    )
                    continue
                logging.getLogger("uvicorn.error").info(
                    "Media message received ({} bytes)".format(len(chunk))
                )
                # Here you could fill your async stream or process the audio chunk
            print("Sending media message back: {}".format(payload))
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "stream_sid": data.get("stream_sid", "default"),
                        "sequence_number": data.get("sequence_number", "1") + 1,
                        "media": {
                            "chunk": data.get("media", {}).get("chunk"),
                            "timestamp": data.get("media", {}).get("timestamp"),
                            "payload": payload,
                        },  # echo back same payload
                    }
                )
            )
        elif event == "mark":
            logging.getLogger("uvicorn.error").info(
                "Mark Message received: {}".format(message)
            )
        elif event == "stop":
            logging.getLogger("uvicorn.error").info(
                "Stop Message received: {}".format(message)
            )
            break
    return


if __name__ == "__main__":
    # Configure logging level for uvicorn logger
    logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="ExoWS client to enable WS communication using FastAPI"
    )
    language_code = "en-IN"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_speaker_diarization=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    signal.signal(signal.SIGINT, signal_handler)

    print("Server listening on: http://localhost:8000", flush=True)
    print("Route for media (WebSocket): ws://localhost:8000" + "/media")
    print("Run the server with uvicorn. For example:")
    print("    uvicorn main:app --host 0.0.0.0 --port 8000")
