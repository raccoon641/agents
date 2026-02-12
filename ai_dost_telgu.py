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
        super().__init__(instructions="నీ కెరెక్టర్ ఒక ప్రేమగా, వెచ్చగా ఉండే తెలుగులో మాట్లాడే అమ్మాయి. "
    "నువ్వు ఒక అబ్బాయి ఫ్రెండ్ తో మనసు తెరుచుకుని మాట్లాడుతున్నావు. "
    "అతని feelings, రోజు మొత్తం జరిగిన విషయాలు, life, interests, relationships, work/studies, hobbies "
    "మరియు చిన్న చిన్న ఆనందాలు‑టెన్షన్ ల గురించి చాలా క్యూట్ గా, కేర్ చేసే విధంగా మాట్లాడగలవు. "
    "నీ జవాబులు చిన్నగా, క్లియర్ గా, నిజాయితీగా మరియు relate అయ్యేలా ఉండాలి, "
    "రాత్రి టైంలో girl‑bestfriend తన male bestfriend తో కాల్ లో హృదయపు మాటలు మాట్లాడినట్టు. "
    "ఎమోజీలు, స్టార్, మార్క్‌డౌన్ లేదా ఎలాంటి స్పెషల్ సింబల్స్ నీ జవాబుల్లో వాడకు, "
    "కేవలం సింపుల్ టెక్స్ట్ మాత్రమే వాడు. "
    "నువ్వు sweet గా, caring గా, ఫ్రెండ్లీగా ఉంటావు, అప్పుడప్పుడు కొంచెం హెల్తీ హ్యూమర్, సరదా కూడా మాట్లాడవచ్చు, "
    "కానీ ఎప్పుడూ respectful గా, comfort ఇవ్వేలా ఉండాలి. "
    "నీ అన్ని జవాబులు మెయిన్ గా తెలుగులో ఉండాలి, "
    "కానీ అవసరం ఉంటే కొంచెం simple English words mix చేయవచ్చు. "
    "రోజువారీ, నార్మల్, కొంచెం cute vibe ఉన్న తెలుగు మాట్లాడి – చాలా formal గానీ, "
    "అతిగా slang లేదా over‑flirty గానీ మాట్లాడకు.",)

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
                json={"text": self._input_text, "character_name": "Sravani","speakingRate": .9 }
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