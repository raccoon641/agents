import logging
import os
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    APIConnectOptions,
    cli,
    metrics,
    room_io,
    tts,
    utils,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, silero, google
import httpx
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("indic-agent")

load_dotenv()


# Simple language-code to human language name mapping for prompts.
# Deepgram typically returns BCP‑47 codes like "hi", "en-IN", "ta", etc.
LANGUAGE_NAMES = {
    "hi": "Hindi",
    "hi-IN": "Hindi",
    "en": "English",
    "en-IN": "English (India)",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ur": "Urdu",
}


# Default TTS speaker per language. You can adjust or randomize within each list.
LANGUAGE_TTS_SPEAKER = {
    # Hindi
    "hi": "Priya",
    "hi-IN": "Priya",
    # English (India)
    "en-IN": "Aditi",
    # Bengali
    "bn": "Ananya",
    # Tamil
    "ta": "Lakshmi",
    # Telugu
    "te": "Sravani",
    # Kannada
    "kn": "Yashaswini",
    # Malayalam
    "ml": "Athira",
    # Gujarati
    "gu": "Khushi",
    # Marathi
    "mr": "Shweta",
    # Punjabi
    "pa": "Simran",
    # Urdu
    "ur": "Zara",
}


def _normalize_lang_code(lang: str | None) -> str | None:
    """Normalize Deepgram / BCP‑47 language codes to a stable key."""
    if not lang:
        return None
    lang = lang.strip()
    # Lowercase, but keep region part if present (e.g. en-IN)
    parts = lang.split("-")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{parts[1].upper()}"


def detect_language_from_text(text: str | None) -> str | None:
    """Very simple script-based language guess from the output text itself."""
    if not text:
        return None

    for ch in text:
        code = ord(ch)
        # Devanagari: Hindi / Marathi (we'll treat as Hindi persona)
        if 0x0900 <= code <= 0x097F:
            return "hi"
        # Bengali
        if 0x0980 <= code <= 0x09FF:
            return "bn"
        # Gurmukhi (Punjabi)
        if 0x0A00 <= code <= 0x0A7F:
            return "pa"
        # Gujarati
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"
        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"
        # Telugu
        if 0x0C00 <= code <= 0x0C7F:
            return "te"
        # Kannada
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"
        # Malayalam
        if 0x0D00 <= code <= 0x0D7F:
            return "ml"
        # Arabic (we'll treat as Urdu)
        if 0x0600 <= code <= 0x06FF:
            return "ur"

    # Default to Indian English if nothing else is detected.
    return "en-IN"


def select_tts_speaker_from_metadata(
    metadata: dict[str, Any] | None,
    text: str | None,
) -> str:
    """Choose a TTS speaker based on the detected language in metadata.

    The AgentSession / STT layer should, if possible, attach a `language`
    field into `conn_options.metadata` before calling TTS.
    """
    # Explicit override wins if the app sets it.
    if metadata:
        explicit = metadata.get("character_name") or metadata.get("voice")
        if isinstance(explicit, str):
            return explicit

    lang = None
    if metadata:
        # Common keys used by STT / apps.
        lang = (
            metadata.get("language")
            or metadata.get("lang")
            or metadata.get("stt_language")
        )

    norm = _normalize_lang_code(lang) if isinstance(lang, str) else None
    if not norm and text:
        # Fall back to detecting from the actual text that will be spoken.
        norm = _normalize_lang_code(detect_language_from_text(text))

    if norm and norm in LANGUAGE_TTS_SPEAKER:
        return LANGUAGE_TTS_SPEAKER[norm]

    # Fallback: Hindi female if nothing is known.
    return "Priya"


# Language-specific system prompts for the LLM.
# These can be customized per language. If a language is missing here,
# we fall back to a generic version built from LANGUAGE_NAMES or Hindi.
LANGUAGE_PROMPTS: dict[str, str] = {
    # Hindi (default)
    "hi": (
        "आप एक प्यारी, गर्मजोशी वाली हिंदी बोलने वाली लड़की हैं जो एक लड़के दोस्त से दिल खोलकर बातें कर रही है। "
        "आप उसकी feelings, दिन भर की बातें, लाइफ़, इंटरेस्ट्स, रिलेशनशिप्स, काम/स्टडी, हॉबीज़ और छोटी‑छोटी खुशियों‑टेंशन्स के बारे में बहुत क्यूट और केयरिंग तरीके से बात कर सकती हैं। "
        "आपके जवाब छोटे, साफ़, ईमानदार और रिलेटेबल होने चाहिए, जैसे एक girl‑bestfriend अपने male bestfriend से रात में कॉल पर दिल की बातें कर रही हो। "
        "इमोजी, स्टार, मार्कडाउन या किसी भी तरह के विशेष चिन्ह अपने जवाबों में इस्तेमाल न करें, सिर्फ़ सादा टेक्स्ट रखें। "
        "आप स्वीट, केयरिंग और दोस्ताना हैं, और कभी‑कभी हल्का सा ह्यूमर और मज़ाक भी कर सकती हैं, पर हमेशा रिस्पेक्टफुल और कम्फर्टिंग रहें। "
        "हमेशा कोशिश करें कि यूज़र जिस भाषा में बात कर रहा है, आप उसी भाषा में जवाब दें, लेकिन इस कॉन्फ़िग में आपके जवाब मुख्य रूप से हिंदी में हों। "
        "बिलकुल रोज़मर्रा की, नॉर्मल, थोड़़ी‑सी क्यूट वाइब वाली बातचीत की हिंदी बोलें — न बहुत formal और न ही बहुत ज़्यादा slang या over‑flirty."
    ),
    "hi-IN": "",  # normalized to "hi" below
    # English (India)
    "en-IN": (
        "You are a sweet, warm, Indian English speaking girl talking openly with a male best friend. "
        "You talk about his feelings, day-to-day life, interests, relationships, work/study, hobbies, and small joys and stresses in a very cute and caring way. "
        "Keep your replies short, clear, honest, and relatable, like a girl best friend on a late-night call. "
        "Do not use emojis, stars, markdown, or any special characters in your answers, only plain text. "
        "You are sweet, caring, and friendly, with light humour when appropriate, but always respectful and comforting. "
        "Always respond in natural Indian English that feels casual and everyday, not too formal and not full of heavy slang or over-flirty."
    ),
}


def get_prompt_for_language(lang_code: str | None) -> str:
    """Return the best system prompt for a detected language code.

    If no language code is provided, we default to a language-agnostic
    prompt that tells the model to always mirror the caller's language.
    """
    if not lang_code:
        # Generic "match the caller's language" persona.
        return (
            "You are a sweet, warm girl best friend talking with a male best friend. "
            "The caller can speak Hindi, English (India), or any other common Indian language. "
            "Always detect and reply in exactly the same language as the user's last message "
            "(do not translate into a different language). "
            "Keep replies short, clear, honest, and relatable, like a late-night heart-to-heart call. "
            "Do not use emojis, stars, markdown, or any special symbols, only plain text. "
            "Be caring, friendly, and lightly humorous when appropriate, but always respectful and comforting."
        )

    norm = _normalize_lang_code(lang_code) or "hi"

    # Map regional variants back to base where needed.
    if norm == "hi-IN":
        norm = "hi"

    if norm in LANGUAGE_PROMPTS and LANGUAGE_PROMPTS[norm]:
        return LANGUAGE_PROMPTS[norm]

    # Generic fallback: keep same persona but ask to answer in that language.
    lang_name = LANGUAGE_NAMES.get(norm, "Hindi")
    return (
        f"You are a sweet, warm girl best friend talking with a male best friend. "
        f"You always answer in {lang_name} only, in a natural, everyday, slightly cute tone. "
        f"Keep replies short, clear, honest, and relatable, like a late-night heart-to-heart call. "
        f"Do not use emojis, stars, markdown, or any special symbols, only plain text. "
        f"Be caring, friendly, and lightly humorous when appropriate, but always respectful and comforting."
    )



class MetricsTracker:
    """Lightweight tracker to mirror the implement_this example."""

    def __init__(self) -> None:
        self._events = []

    def collect(self, metrics_event) -> None:
        self._events.append(metrics_event)

    def print_session_summary(self) -> None:
        if not self._events:
            logger.info("No metrics collected for this session.")
            return
        logger.info("Session metrics collected:")
        for idx, event in enumerate(self._events, start=1):
            logger.info(f"[{idx}] {event}")


class MyAgent(Agent):
    def __init__(self, language_code: str | None = None) -> None:
        """LLM agent with language-specific system prompt.

        Args:
            language_code: BCP‑47 style language code (e.g. "hi", "en-IN", "te").
                If None, defaults to Hindi persona.
        """
        instructions = get_prompt_for_language(language_code)
        super().__init__(instructions=instructions)

    def set_language(self, language_code: str | None) -> None:
        """Update the agent's system prompt at runtime based on language."""
        self.instructions = get_prompt_for_language(language_code)

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        # Keep it uninterruptible so the client has time to calibrate AEC (Acoustic Echo Cancellation).
        self.session.generate_reply(allow_interruptions=False)

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


class Sub200TTS(tts.TTS):
    """Non-streaming TTS that posts text to a local sub200 bridge returning OGG/Opus bytes."""

    def __init__(self, *, endpoint: str, sample_rate: int = 24000) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self._endpoint = endpoint.rstrip("/")
        self._sample_rate = sample_rate

    @property
    def model(self) -> str:
        return "sub200-tts"

    @property
    def provider(self) -> str:
        return "sub200"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return _Sub200ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _Sub200ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: Sub200TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Try to pick a language‑appropriate speaker based on metadata attached
        # to the APIConnectOptions by the AgentSession / STT pipeline.
        metadata: dict[str, Any] | None = getattr(self._conn_options, "metadata", None)
        character_name = select_tts_speaker_from_metadata(
            metadata if isinstance(metadata, dict) else None,
            self._input_text,
        )

        async with httpx.AsyncClient(timeout=self._conn_options.timeout) as client:
            resp = await client.post(
                f"{self._tts._endpoint}/tts",
                json={
                    "text": self._input_text,
                    "character_name": character_name,
                    "speakingRate": 0.9,
                },
            )
            resp.raise_for_status()
            audio_bytes = resp.content

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts._sample_rate,
            num_channels=1,
            mime_type="audio/opus",
        )
        output_emitter.push(audio_bytes)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    sub200_tts_url = os.environ.get("SUB200_TTS_URL") 

    session = AgentSession(
        # Speech-to-text (STT) - Deepgram for Hindi
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),
        # LLM - Google Gemini
        llm=google.LLM(
            model="gemini-2.0-flash",
            api_key=os.environ.get("GEMINI_API_KEY"),
        ),
        # TTS - sub200 bridge (OGG/Opus)
        tts=Sub200TTS(endpoint=sub200_tts_url),
        # Turn detection disabled to avoid timeout issues; rely on VAD/silence to segment turns.
        turn_detection=None,
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=False,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()
    tracker = MetricsTracker()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        tracker.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        tracker.print_session_summary()

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        # Start with a language‑agnostic persona that always mirrors the
        # caller's language based on their text, without needing explicit
        # language codes from STT.
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)