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

def get_groq_completion(transcript="Hello"):
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
    breakpoint()

    return completion.choices[0].message.content

get_groq_completion()