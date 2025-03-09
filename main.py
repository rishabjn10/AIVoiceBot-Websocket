import argparse
import asyncio
import base64
import io
import json
import logging
import os
import signal
import sys
import threading
import wave

import janus  # pip install janus
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1p1beta1 as speech
from groq import Groq
from gtts import gTTS
from pydub import AudioSegment

load_dotenv(override=True)

app = FastAPI()

# Audio recording parameters
RATE = 8000
CHUNK = int(RATE / 10)  # 100ms

# Global list for saving audio file and a thread-safe queue for live transcription.
audio_chunks = []
# Create two thread-safe queues:
# 1. For incoming audio chunks (for live transcription)
incoming_audio_queue = janus.Queue()
# 2. For outgoing response objects (to send back transcripts with TTS audio)
incoming_metadata_queue = janus.Queue()

# Initialize Speech client and configuration.
client = speech.SpeechClient()
language_code = "hi-IN"  # Set language code to Hindi
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code=language_code,
    enable_speaker_diarization=True,
)
streaming_config = speech.StreamingRecognitionConfig(
    config=config, interim_results=False
)
# We have some additional configuration options for streaming_config:enable_voice_activity_events : bool
# If true, responses with voice activity speech events will be returned as they are detected.

# voice_activity_timeout : google.cloud.speech_v1p1beta1.types.StreamingRecognitionConfig.VoiceActivityTimeout
# If set, the server will automatically close the stream after the specified duration has elapsed after the last VOICE_ACTIVITY speech event has been sent. The field voice_activity_events must also be set to true.


def signal_handler(sig, frame):
    sys.exit(0)


def create_response_voice(transcription):
    # Generate TTS as an MP3 stream
    tts = gTTS(text=transcription, lang="en")
    tts_fp = io.BytesIO()
    tts.write_to_fp(tts_fp)
    tts_fp.seek(0)

    # Use pydub to load the MP3 and convert it
    audio_segment = AudioSegment.from_file(tts_fp, format="mp3")
    # Convert to the desired format:
    #  - 8kHz frame rate
    #  - mono channel (1 channel)
    #  - 16-bit samples (2 bytes per sample)
    audio_segment = (
        audio_segment.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    )

    # Get the raw PCM bytes (this is in little-endian by default)
    pcm_data = audio_segment.raw_data

    # Encode the PCM bytes in base64 to send as payload
    return base64.b64encode(pcm_data).decode("utf-8")


def get_groq_completion(transcript=""):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system_prompt = "No matter what, always return your response in hindi."
    completion = client.chat.completions.create(
        model="qwen-2.5-32b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": transcript},
        ],
        temperature=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content


def listen_print_loop(responses, loop, websocket):
    """Iterate through recognition responses and print transcriptions."""
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        print("User said this: ", transcript)
        ai_response = get_groq_completion(transcript)
        print("AI said this: ", ai_response)

        # TODO: Implement RAG for better response for domain specific queries
        b64_audio = create_response_voice(ai_response)
        metadata = incoming_metadata_queue.sync_q.get_nowait()
        message = json.dumps(
            {
                "event": "media",
                "stream_sid": metadata.get("stream_sid", "default"),
                "sequence_number": str(int(metadata.get("sequence_number", "1")) + 1),
                "media": {
                    "chunk": metadata.get("media", {}).get("chunk"),
                    "timestamp": metadata.get("media", {}).get("timestamp"),
                    "payload": b64_audio,
                },
            }
        )
        # Schedule sending the transcript on the main event loop
        asyncio.run_coroutine_threadsafe(websocket.send_text(message), loop)


def generator_wrapper():
    """
    Synchronous generator that pulls audio chunks from the thread-safe queue.
    Blocks on incoming_audio_queue.sync_q.get(), and stops when it gets None.
    """
    while True:
        chunk = (
            incoming_audio_queue.sync_q.get()
        )  # Blocking call in the background thread.
        if chunk is None:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)


def start_recognition_loop(loop, websocket):
    """
    Run the streaming recognition in a background thread.
    This function uses a synchronous generator that pulls data from incoming_audio_queue.
    """
    responses = client.streaming_recognize(streaming_config, generator_wrapper())
    listen_print_loop(responses, loop, websocket)


@app.websocket("/media")
async def media_socket(websocket: WebSocket):
    await websocket.accept()
    logging.getLogger("uvicorn.error").info("WebSocket connection accepted")

    # Get the current event loop
    loop = asyncio.get_running_loop()

    # Start the background recognition thread (if not already running)
    recognition_thread = threading.Thread(
        target=start_recognition_loop, args=(loop, websocket), daemon=True
    )
    recognition_thread.start()

    try:
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
            if event == "media":
                await incoming_metadata_queue.async_q.put(data)
                payload = data.get("media", {}).get("payload")
                if payload:
                    try:
                        # Decode the base64-encoded audio chunk.
                        chunk = base64.b64decode(payload)
                        # Save chunk for later file writing.
                        audio_chunks.append(chunk)
                        # Place chunk into the thread-safe queue for live transcription.
                        await incoming_audio_queue.async_q.put(chunk)
                    except Exception as e:
                        logging.getLogger("uvicorn.error").error(
                            "Error decoding payload: {}".format(e)
                        )
                        continue
            # Handle other events as needed.
    finally:
        # When the connection closes, write out the full audio file.
        output_filename = "output.wav"
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio = 2 bytes per sample
            wf.setframerate(RATE)  # 8kHz sample rate
            wf.writeframes(b"".join(audio_chunks))
        logging.getLogger("uvicorn.error").info(
            "Audio saved to '{}'".format(output_filename)
        )
        # Signal the generator to exit.
        await incoming_audio_queue.async_q.put(None)


if __name__ == "__main__":
    logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(
        description="ExoWS client to enable WS communication using FastAPI"
    )
    signal.signal(signal.SIGINT, signal_handler)

    print("Server listening on: http://localhost:8000", flush=True)
    print("Route for media (WebSocket): ws://localhost:8000/media")
    print("Run the server with uvicorn. For example:")
    print("    uvicorn main:app --host 0.0.0.0 --port 8000")
