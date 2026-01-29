import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, silero, openai, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.tts.stream_adapter import StreamAdapter

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
        # TTS - Google Cloud TTS (Hindi, female Neural2 voice, non-streaming)
        # Streaming synthesis currently only supports Chirp3 HD voices; to use Hindi Neural2
        # reliably we disable streaming and use standard synthesize_speech.
        tts=google.TTS(
            language="hi-IN",
            gender="female",
            voice_name="hi-IN-Neural2-A",
            speaking_rate=1.0,
            use_streaming=False,
            credentials_file=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
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
