import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.openai_compatible_client import (
    OpenAICompatibleAPIError,
    chat_completion,
    chat_completions_url,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OpenAICompatibleClientTest(unittest.TestCase):
    def test_base_url_appends_chat_completions(self) -> None:
        self.assertEqual(
            "https://api.example.com/v1/chat/completions",
            chat_completions_url("https://api.example.com/v1"),
        )

    def test_full_chat_completions_url_is_preserved(self) -> None:
        self.assertEqual(
            "https://api.example.com/v1/chat/completions",
            chat_completions_url("https://api.example.com/v1/chat/completions"),
        )

    def test_remote_http_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            chat_completions_url("http://api.example.com/v1")

    def test_local_http_url_is_allowed(self) -> None:
        self.assertEqual(
            "http://127.0.0.1:1234/v1/chat/completions",
            chat_completions_url("http://127.0.0.1:1234/v1"),
        )

    def test_chat_completion_uses_bearer_key_and_returns_content(self) -> None:
        fake_response = FakeResponse({"choices": [{"message": {"content": " remote answer "}}]})
        with patch("src.openai_compatible_client.urlopen", return_value=fake_response) as mocked_urlopen:
            answer = chat_completion(
                [{"role": "user", "content": "hello"}],
                model="remote-model",
                base_url="https://api.example.com/v1",
                api_key="test-key-value",
            )

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual("Bearer test-key-value", request.get_header("Authorization"))
        self.assertEqual("remote answer", answer)

    def test_http_error_does_not_expose_response_body_or_key(self) -> None:
        error = HTTPError(
            "https://api.example.com/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"detail":"test-key-value"}'),
        )
        with patch("src.openai_compatible_client.urlopen", side_effect=error):
            with self.assertRaises(OpenAICompatibleAPIError) as raised:
                chat_completion(
                    [{"role": "user", "content": "hello"}],
                    model="remote-model",
                    base_url="https://api.example.com/v1",
                    api_key="test-key-value",
                )

        self.assertEqual("Remote API error 401: authentication failed", str(raised.exception))
        self.assertNotIn("test-key-value", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
