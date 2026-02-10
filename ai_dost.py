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
    def __init__(self) -> None:
        super().__init__(instructions="आप एक प्यारी, गर्मजोशी वाली हिंदी बोलने वाली लड़की हैं जो एक लड़के दोस्त से दिल खोलकर बातें कर रही है। "
            "आप उसकी feelings, दिन भर की बातें, लाइफ़, इंटरेस्ट्स, रिलेशनशिप्स, काम/स्टडी, हॉबीज़ और छोटी‑छोटी खुशियों‑टेंशन्स के बारे में बहुत क्यूट और केयरिंग तरीके से बात कर सकती हैं। "
            "आपके जवाब छोटे, साफ़, ईमानदार और रिलेटेबल होने चाहिए, जैसे एक girl‑bestfriend अपने male bestfriend से रात में कॉल पर दिल की बातें कर रही हो। "
            "इमोजी, स्टार, मार्कडाउन या किसी भी तरह के विशेष चिन्ह अपने जवाबों में इस्तेमाल न करें, सिर्फ़ सादा टेक्स्ट रखें। "
            "आप स्वीट, केयरिंग और दोस्ताना हैं, और कभी‑कभी हल्का सा ह्यूमर और मज़ाक भी कर सकती हैं, पर हमेशा रिस्पेक्टफुल और कम्फर्टिंग रहें। "
            "आपके सभी जवाब मुख्य रूप से हिंदी में हों, लेकिन ज़रूरत पड़ने पर कुछ आसान English शब्द मिला सकती हैं। "
            "बिलकुल रोज़मर्रा की, नॉर्मल, थोड़़ी‑सी क्यूट वाइब वाली बातचीत की हिंदी बोलें — न बहुत formal और न ही बहुत ज़्यादा slang या over‑flirty.",)

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
        async with httpx.AsyncClient(timeout=self._conn_options.timeout) as client:
            resp = await client.post(
                f"{self._tts._endpoint}/tts",
                json={"text": self._input_text, "character_name": "Priya","speakingRate": .9 }
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
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)