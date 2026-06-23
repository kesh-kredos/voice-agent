import numpy as np
from scipy.signal import resample_poly
import base64
import json

def parse_twilio_media(msg: str) -> bytes | None:
    try:
        data = json.loads(msg)
        if data.get('event') == 'media':
            return base64.b64decode(data['media']['payload'])
        return None
    except (json.JSONDecodeError, KeyError):
        return None
    
def twilio_media_message(audio: bytes, stream_sid: str) -> str:
    return json.dumps({
        'event': 'media',
        'streamSid': stream_sid,
        'media': {
            'payload': base64.b64encode(audio).decode('utf-8')
        }
    })


def mulaw_to_float32(mulaw_bytes: bytes) -> np.ndarray:
    pcm16 = _mulaw_to_linear(np.frombuffer(mulaw_bytes, dtype=np.uint8))
    upsampled = resample_poly(pcm16, up=2, down=1)
    return upsampled.astype(np.float32) / 32768.0


def float32_to_mulaw(audio: np.ndarray) -> bytes:
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

def _linear_to_mulaw(pcm: np.ndarray) -> np.ndarray:
    BIAS = 33
    pcm = pcm.astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0x00)
    pcm = np.abs(pcm)
    pcm = np.clip(pcm, 0, 32635) + BIAS
    exp = np.clip(np.floor(np.log2(pcm)).astype(np.int32) - 5, 0, 7)
    mantissa = (pcm >> (exp + 1)) & 0x0F

    return (~(sign | (exp << 4) | mantissa)).astype(np.uint8)