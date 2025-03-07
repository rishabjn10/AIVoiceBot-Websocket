# Audio Chat WebSocket Server

This project is a real-time audio processing and conversational AI server built using FastAPI. It streams incoming audio via a WebSocket, transcribes it using Google Cloud Speech-to-Text, processes the transcription through the Groq conversational AI API, and converts the AI’s response back into speech using gTTS. Finally, it sends the synthesized audio back to the client while also saving the complete audio stream to a file.

## Features

- **Real-Time Audio Streaming:** Processes audio chunks received over a WebSocket connection.
- **Speech Recognition:** Utilizes Google Cloud Speech-to-Text (configured for Hindi) to transcribe audio.
- **Conversational AI Integration:** Leverages the Groq API (using the Qwen model) to generate responses in Hindi.
- **Text-to-Speech (TTS):** Converts the AI response into speech with gTTS and processes audio with pydub.
- **Thread-Safe Queues:** Uses the Janus library for managing asynchronous and synchronous queues.
- **Audio File Saving:** Saves the complete session’s audio stream as `output.wav` when the WebSocket connection closes.

## Prerequisites

- **Python:** 3.9 or later.
- **pdm:** For dependency management. ([Installation Guide](https://pdm.fming.dev/latest/))
- **Google Cloud Speech-to-Text Credentials:** Set up and configured. You can set the path to your credentials via the environment variable `GOOGLE_APPLICATION_CREDENTIALS`.
- **Groq API Key:** A valid API key for accessing the Groq API.
- Other dependencies listed in the project (see below).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/rishabjn10/AIVoiceBot-Websocket.git
   cd AIVoiceBot-Websocket
   ```

2. **Install Dependencies with pdm:**

   ```bash
   pdm install
   ```

3. **Environment Variables:**

   Create a `.env` file in the project root with your configuration:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google/credentials.json
   ```

## Running the Server

Start the server using the following command:

```bash
pdm run uvicorn mainC:app --host 0.0.0.0 --port 8000
```

- **HTTP Endpoint:** [http://localhost:8000](http://localhost:8000)
- **WebSocket Endpoint:** `ws://localhost:8000/media`

## How It Works

1. **Audio Reception:**  
   The server accepts a WebSocket connection on the `/media` route. Clients send base64-encoded audio chunks along with metadata.

2. **Speech Recognition:**  
   Audio chunks are queued and sent to the Google Cloud Speech-to-Text API for transcription.

3. **Conversational AI:**  
   The transcribed text is sent to the Groq API, which returns a response. The system prompt forces responses in Hindi.

4. **Text-to-Speech Conversion:**  
   The AI’s text response is converted into speech via gTTS, processed to meet the audio format requirements (8kHz, mono, 16-bit), and then encoded in base64.

5. **Response Delivery:**  
   The TTS audio is sent back to the client over the same WebSocket connection.

6. **Audio File Saving:**  
   When the WebSocket connection closes, the complete audio stream is saved as `output.wav`.

## File Structure Overview

- **`mainC.py`**  
  Contains the FastAPI application, WebSocket handling logic, audio processing routines, and integration with external APIs (Google Cloud Speech, Groq, gTTS).

- **Environment Setup:**  
  Utilizes `python-dotenv` for managing environment variables.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
- [Groq API](https://groq.com/)
- [gTTS](https://pypi.org/project/gTTS/)
- [pydub](https://github.com/jiaaro/pydub)
- [Janus](https://github.com/aio-libs/janus)