# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace
from typing import Final

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

SUB200_BASE_URL: Final[str] = "http://tts.sub200.dev/indic-19/v1/tts/generate"
NUM_CHANNELS: Final[int] = 1
MIME_TYPE: Final[str] = "audio/wav"


@dataclass
class _TTSOptions:
    base_url: str
    voice: str
    sample_rate: int
    num_channels: int


class TTS(tts.TTS):
    """
    Text-to-Speech (TTS) plugin for Sub200 Indic TTS.

    This plugin provides text-to-speech synthesis using Sub200's Indic TTS API.
    It supports various Indian languages and voices.

    Args:
        voice: The voice ID to use. Default is "Kishan".
        sample_rate: Audio sample rate in Hz. Default is 8000.
        num_channels: Number of audio channels. Default is 1 (mono).
        base_url: Custom base URL for the Sub200 TTS API. Defaults to the standard endpoint.
    """

    def __init__(
        self,
        *,
        voice: str = "Kishan",
        sample_rate: int = 8000,
        num_channels: int = 1,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._opts = _TTSOptions(
            base_url=base_url if is_given(base_url) else SUB200_BASE_URL,
            voice=voice,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "indic-19"

    @property
    def provider(self) -> str:
        return "Sub200"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice: The voice ID to update.
            sample_rate: Output sample rate in Hz.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            self._sample_rate = sample_rate

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def aclose(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text to speech in chunks."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        data = {
            "voice": self._opts.voice,
            "text": self._input_text,
            "stream": True,
            "sample_rate": self._opts.sample_rate,
        }

        try:
            async with self._tts._ensure_session().post(
                self._opts.base_url,
                headers={
                    "Content-Type": "application/json",
                },
                json=data,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                # Check if response is audio
                content_type = resp.headers.get("Content-Type", "")
                if not content_type.startswith("audio/"):
                    content = await resp.text()
                    raise APIStatusError(
                        message=f"Sub200 TTS API returned non-audio response: {content}",
                        status_code=resp.status,
                        request_id=None,
                        body=content,
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=MIME_TYPE,
                )

                async for chunk, _ in resp.content.iter_chunks():
                    if chunk:
                        output_emitter.push(chunk)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

