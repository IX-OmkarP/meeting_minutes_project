import unittest

from transcription_utils import choose_chat_model


class ChooseChatModelTests(unittest.TestCase):
    def test_prefers_best_available_model(self):
        available = ["whisper-large-v3", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        self.assertEqual(choose_chat_model(available), "llama-3.3-70b-versatile")

    def test_skips_models_the_key_cannot_use(self):
        available = ["whisper-large-v3", "llama-3.1-8b-instant"]
        self.assertEqual(choose_chat_model(available), "llama-3.1-8b-instant")

    def test_falls_back_to_any_non_audio_model(self):
        available = ["whisper-large-v3", "some-new-chat-model", "playai-tts"]
        self.assertEqual(choose_chat_model(available), "some-new-chat-model")

    def test_env_override_wins(self):
        self.assertEqual(
            choose_chat_model(["llama-3.3-70b-versatile"], override="my-model"),
            "my-model",
        )

    def test_raises_when_only_audio_models_exist(self):
        with self.assertRaises(RuntimeError):
            choose_chat_model(["whisper-large-v3", "playai-tts"])


if __name__ == "__main__":
    unittest.main()
