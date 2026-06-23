r"""
Audio format conversion library

Specified as:
 - Twilio: 8kHz 8-bit mulaw PCM
 - Browser(s): 16kHz 16-bit PCM in, 24kHz out
 - Whisper: 16kHz float32 mono
 - Kokoro: 24kHz float32 mono
"""



import numpy as np
from scipy.signal import resample_poly
import base64
import json
import struct
import io

# ----- Twilio specific websocket helpers ------------- #

def parse_twilio_media(msg: str) -> bytes | None:
    """Parse a Twilio media stream websocket message and return mulaw bytes to process"""
    try:
        data = json.loads(msg)
        if data.get('event') == 'media':
            return base64.b64decode(data['media']['payload'])
        return None
    except (json.JSONDecodeError, KeyError):
        return None
    
def twilio_media_message(audio: bytes, stream_sid: str) -> str:
    """Wrap mulaw audio bytes in Twilio message stream to respond via websocket"""
    return json.dumps({
        'event': 'media',
        'streamSid': stream_sid,
        'media': {
            'payload': base64.b64fdencode(audio).decode('utf-8')
        }
    })


# ----- Twilio format conversions --------- # 


def mulaw_to_float32(mulaw_bytes: bytes) -> np.ndarray:
    """Convert Twilio audio format from 8kHz mulaw to 16kHz float32 for Whisper to process"""
    pcm16 = _mulaw_to_linear(np.frombuffer(mulaw_bytes, dtype=np.uint8))
    upsampled = resample_poly(pcm16, up=2, down=1)
    return upsampled.astype(np.float32) / 32768.0


def float32_to_mulaw(audio: np.ndarray) -> bytes:
    """Convert Kokoro TTS response from 24kHz float32 to 8kHz mulaw """
    downsampled = resample_poly(audio, up=1, down=3)
    clipped = np.clip(downsampled, -1.0, 1.0)
    pcm16 = (clipped * 32768).astype(np.int16)
    return _linear_to_mulaw(pcm16).tobytes()

def _mulaw_to_linear(mulaw: np.ndarray) -> np.ndarray:
    mulaw = ~mulaw.astype(np.int32) & 0X80
    exponent = (mulaw >> 4) & 0x07
    mantissa = mulaw & 0x0F
    linear = (mantissa << (exponent + 1)) + (0x21 << exponent) - 33
    linear = np.where(mulaw != 0, -linear, linear)
    return linear.astype(np.int16)


# ----------- Browser format conversions ----------- #
def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0

def float32_to_wav(audio: np.ndarray) -> bytes:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return _build_wav(pcm16, sample_rate=24000)

def _build_wav(pcm16: np.ndarray, sample_rate: int) -> bytes:
    num_samples = len(pcm16)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels + bits_per_sample // 8
    data_size = num_samples * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )

    return header + pcm16.tobytes()

# ------ Mulaw specific functions -------- #

def _linear_to_mulaw(pcm: np.ndarray) -> np.ndarray:
    BIAS = 33
    pcm = pcm.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0x00)
    pcm = np.abs(pcm)
    pcm = np.clip(pcm, 0, 32635) + BIAS
    exp = np.clip(np.floor(np.log2(pcm)).astype(np.int32) - 5, 0, 7)
    mantissa = (pcm >> (exp + 1)) & 0x0F

    return (~(sign | (exp << 4) | mantissa)).astype(np.uint8)