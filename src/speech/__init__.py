from .speech_recognition import (
    SpeechRecognizer,
    HealthcareSpeechRecognizer,
    SpeechRecognitionEngine
)

from .speech_synthesis import (
    SpeechSynthesizer,
    HealthcareSpeechSynthesizer,
    TextToSpeechEngine
)

from .voice_assistant import (
    VoiceAssistant,
    HealthcareVoiceAssistant,
    VoiceInteractionManager
)

__version__ = "1.0.0"

def create_healthcare_voice_assistant() -> HealthcareVoiceAssistant:
    """Create a pre-configured healthcare voice assistant."""
    return HealthcareVoiceAssistant()

def create_speech_engine() -> SpeechRecognitionEngine:
    """Create a speech recognition engine."""
    return SpeechRecognitionEngine()

def create_tts_engine() -> TextToSpeechEngine:
    """Create a text-to-speech engine."""
    return TextToSpeechEngine()

__all__ = [
    'SpeechRecognizer',
    'HealthcareSpeechRecognizer',
    'SpeechRecognitionEngine',
    'SpeechSynthesizer',
    'HealthcareSpeechSynthesizer', 
    'TextToSpeechEngine',
    'VoiceAssistant',
    'HealthcareVoiceAssistant',
    'VoiceInteractionManager',
    'create_healthcare_voice_assistant',
    'create_speech_engine',
    'create_tts_engine'
]
