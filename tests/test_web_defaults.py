import os
import unittest
from unittest.mock import patch

from webapp.server import (
    build_routing_plan,
    default_llm_provider,
    default_planned_fusion_mode,
    default_retrieval_mode,
    generate_with_provider,
    handle_provider_test,
    mark_generation_repaired,
    should_fallback_to_ollama,
)


class WebDefaultsTest(unittest.TestCase):
    def test_llm_provider_defaults_to_local_ollama(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("webapp.server.get_deepseek_api_key", return_value="key"):
            self.assertEqual(default_llm_provider(), "ollama")

    def test_explicit_deepseek_provider_requires_key(self) -> None:
        with patch.dict(os.environ, {"RAG_DEFAULT_LLM_PROVIDER": "deepseek"}, clear=True):
            with patch("webapp.server.get_deepseek_api_key", return_value=None):
                self.assertEqual(default_llm_provider(), "ollama")
            with patch("webapp.server.get_deepseek_api_key", return_value="key"):
                self.assertEqual(default_llm_provider(), "deepseek")

    def test_retrieval_defaults_to_direct(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_retrieval_mode(), "direct")

    def test_valid_explicit_mode_is_preserved(self) -> None:
        with patch.dict(os.environ, {"RAG_DEFAULT_RETRIEVAL_MODE": "AUTO"}, clear=True):
            self.assertEqual(default_retrieval_mode(), "auto")

    def test_invalid_explicit_mode_falls_back_to_direct(self) -> None:
        with patch.dict(os.environ, {"RAG_DEFAULT_RETRIEVAL_MODE": "unknown"}, clear=True):
            self.assertEqual(default_retrieval_mode(), "direct")

    def test_planned_fusion_defaults_to_conservative_v3(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_planned_fusion_mode(), "conservative")

    def test_legacy_planned_fusion_can_be_selected_for_ab(self) -> None:
        with patch.dict(os.environ, {"RAG_PLANNED_FUSION_MODE": "legacy"}, clear=True):
            self.assertEqual(default_planned_fusion_mode(), "legacy")

    def test_anchored_planned_fusion_can_be_selected_for_ab(self) -> None:
        with patch.dict(os.environ, {"RAG_PLANNED_FUSION_MODE": "anchored"}, clear=True):
            self.assertEqual(default_planned_fusion_mode(), "anchored")

    def test_invalid_planned_fusion_falls_back_to_conservative(self) -> None:
        with patch.dict(os.environ, {"RAG_PLANNED_FUSION_MODE": "unknown"}, clear=True):
            self.assertEqual(default_planned_fusion_mode(), "conservative")

    def test_temporary_remote_provider_cannot_become_environment_default(self) -> None:
        with patch.dict(os.environ, {"RAG_DEFAULT_LLM_PROVIDER": "openai_compatible"}, clear=True):
            self.assertEqual(default_llm_provider(), "ollama")

    def test_conservative_fusion_uses_planner_v3_shape(self) -> None:
        plan = build_routing_plan("检索评估时召回率和 citation 指标应该怎么组合？", "conservative")

        self.assertEqual("rules_v3_conservative", plan.planner_version)

    def test_anchored_fusion_keeps_legacy_planner_shape(self) -> None:
        plan = build_routing_plan("检索评估时召回率和 citation 指标应该怎么组合？", "anchored")

        self.assertNotEqual("rules_v3_conservative", plan.planner_version)

    def test_policy_errors_fallback_to_ollama(self) -> None:
        error = RuntimeError(
            "Invalid prompt: your prompt was flagged as potentially violating our usage policy."
        )
        self.assertTrue(should_fallback_to_ollama(error))
        self.assertFalse(should_fallback_to_ollama(RuntimeError("network timeout")))

    def test_deepseek_policy_error_uses_ollama_fallback(self) -> None:
        with patch(
            "webapp.server.chat_completion",
            side_effect=RuntimeError("Invalid prompt: prompt was flagged"),
        ), patch("webapp.server.generate", return_value="local answer"):
            answer, generation = generate_with_provider(
                "prompt",
                llm_provider="deepseek",
                ollama_model="qwen2.5:1.5b",
                ollama_host="http://127.0.0.1:11434",
                deepseek_model="deepseek-v4-flash",
                deepseek_base_url="https://api.deepseek.com",
            )
        self.assertEqual(answer, "local answer")
        self.assertEqual(generation["provider"], "ollama")
        self.assertTrue(generation["fallback_used"])
        self.assertEqual(
            generation["provider_path"],
            ["deepseek:deepseek-v4-flash", "ollama:qwen2.5:1.5b"],
        )
        self.assertNotIn("cloud_error", generation)

    def test_empty_deepseek_answer_uses_ollama_fallback(self) -> None:
        with patch("webapp.server.chat_completion", return_value=""), patch(
            "webapp.server.generate", return_value="local answer"
        ):
            answer, generation = generate_with_provider(
                "prompt",
                llm_provider="deepseek",
                ollama_model="qwen2.5:1.5b",
                ollama_host="http://127.0.0.1:11434",
                deepseek_model="deepseek-chat",
                deepseek_base_url="https://api.deepseek.com",
            )
        self.assertEqual(answer, "local answer")
        self.assertEqual("cloud provider returned empty content", generation["fallback_reason"])

    def test_openai_compatible_provider_uses_temporary_credentials_without_returning_them(self) -> None:
        with patch("webapp.server.openai_compatible_chat_completion", return_value="remote answer") as remote_call:
            answer, generation = generate_with_provider(
                "prompt",
                llm_provider="openai_compatible",
                ollama_model="qwen2.5:1.5b",
                ollama_host="http://127.0.0.1:11434",
                deepseek_model="deepseek-chat",
                deepseek_base_url="https://api.deepseek.com",
                remote_api_model="remote-model",
                remote_api_base_url="https://api.example.com/v1",
                remote_api_key="test-key-value",
            )

        self.assertEqual("remote answer", answer)
        self.assertEqual("openai_compatible", generation["provider"])
        self.assertEqual(["remote:remote-model"], generation["provider_path"])
        self.assertNotIn("test-key-value", repr(generation))
        self.assertEqual("test-key-value", remote_call.call_args.kwargs["api_key"])

    def test_provider_connection_result_omits_url_and_key(self) -> None:
        payload = {
            "remote_api_model": "remote-model",
            "remote_api_base_url": "https://api.example.com/v1",
            "remote_api_key": "test-key-value",
        }
        with patch("webapp.server.openai_compatible_chat_completion", return_value="OK"):
            result = handle_provider_test(payload)

        self.assertTrue(result["ok"])
        self.assertEqual("remote-model", result["model"])
        self.assertNotIn("remote_api_key", result)
        self.assertNotIn("remote_api_base_url", result)
        self.assertNotIn("test-key-value", repr(result))

    def test_repaired_answer_records_the_final_provider_path(self) -> None:
        generation = {
            "requested_provider": "ollama",
            "provider": "ollama",
            "provider_path": ["ollama:qwen2.5:1.5b"],
            "fallback_used": False,
        }

        repaired = mark_generation_repaired(generation, "deepseek-chat")

        self.assertEqual("deepseek", repaired["provider"])
        self.assertTrue(repaired["repair_used"])
        self.assertEqual(
            ["ollama:qwen2.5:1.5b", "deepseek-repair:deepseek-chat"],
            repaired["provider_path"],
        )


if __name__ == "__main__":
    unittest.main()
