# Sub200 Indic TTS Plugin for LiveKit Agents

This plugin provides text-to-speech synthesis using Sub200's Indic TTS API, supporting various Indian languages and voices.

## Installation

```bash
pip install livekit-plugins-sub200
```

## Usage

```python
from livekit.agents import llm
from livekit.plugins import sub200

# Initialize the TTS
tts = sub200.TTS(
    voice="Kishan",  # Default voice
    sample_rate=8000,  # Default sample rate
)

# Use with an agent
agent = llm.LLMAgent(
    tts=tts,
    # ... other options
)
```

## Configuration

### Parameters

- `voice` (str): The voice ID to use. Default is `"Kishan"`.
- `sample_rate` (int): Audio sample rate in Hz. Default is `8000`.
- `num_channels` (int): Number of audio channels. Default is `1` (mono).
- `base_url` (str, optional): Custom base URL for the Sub200 TTS API. Defaults to the standard endpoint.

### Example

```python
from livekit.plugins import sub200

# Create TTS instance with custom voice
tts = sub200.TTS(
    voice="Kishan",
    sample_rate=8000,
)

# Update options dynamically
tts.update_options(voice="AnotherVoice")
```

## API Details

The plugin uses the Sub200 Indic TTS API endpoint:
- **Endpoint**: `http://tts.sub200.dev/indic-19/v1/tts/generate`
- **Method**: POST
- **Content-Type**: application/json
- **Response**: WAV audio file

### Request Format

```json
{
    "voice": "Kishan",
    "text": "नमस्ते! आपका स्वागत है।",
    "stream": true,
    "sample_rate": 8000
}
```

## License

Apache-2.0

