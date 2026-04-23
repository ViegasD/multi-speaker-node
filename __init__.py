from .multispeaker_lipsync import MultiSpeakerLipSync

NODE_CLASS_MAPPINGS = {
    "MultiSpeakerLipSync": MultiSpeakerLipSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiSpeakerLipSync": "Multi-Speaker LipSync (ElevenLabs + LatentSync)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
