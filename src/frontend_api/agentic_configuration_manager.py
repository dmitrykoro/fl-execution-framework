from __future__ import annotations

import asyncio
import json
import time
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.ai_agent_wrapper import AIAgent
from src.utils.common import load_json_as_str


@dataclass
class AgentChatMessage:
    role: str
    text: str


@dataclass
class AgentSession:
    id: str
    vendor: Any
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 0
    phase: str = "idle"
    pending: bool = False
    error: str | None = None
    user_prompt: str | None = None
    latest_config_text: str | None = None
    latest_config_path: str | None = None
    latest_reasoning: str | None = None
    num_corrections: int = 0
    run_id: str | None = None
    result_summary: str | None = None
    result_summary_run_id: str | None = None
    messages: list[AgentChatMessage] = field(default_factory=list)
    update_event: asyncio.Event = field(default_factory=asyncio.Event)


class AgenticConfigurationManager:
    def __init__(self, ai_agent_config_filename: str = "ai_agent_config.json") -> None:
        self.ai_agent_config_filename = ai_agent_config_filename
        self.sessions: dict[str, AgentSession] = {}

    def _new_vendor(self):
        agent = AIAgent(ai_agent_config_filename=self.ai_agent_config_filename)
        return agent.ai_vendor_api

    def _bump(self, session: AgentSession) -> None:
        session.version += 1
        session.updated_at = time.time()
        session.update_event.set()

    def _snapshot(self, session: AgentSession) -> dict[str, Any]:
        return {
            "session_id": session.id,
            "version": session.version,
            "phase": session.phase,
            "pending": session.pending,
            "error": session.error,
            "user_prompt": session.user_prompt,
            "latest_config_text": session.latest_config_text,
            "latest_config_path": session.latest_config_path,
            "latest_reasoning": session.latest_reasoning,
            "num_corrections": session.num_corrections,
            "run_id": session.run_id,
            "result_summary": session.result_summary,
            "result_summary_run_id": session.result_summary_run_id,
            "messages": [{"role": message.role, "text": message.text} for message in session.messages],
        }

    def _append_message(self, session: AgentSession, role: str, text: str) -> None:
        session.messages.append(AgentChatMessage(role=role, text=text))

    def _prepare_final_config_metadata(self, session: AgentSession) -> None:
        try:
            session.vendor.update_final_generated_config(
                user_prompt=session.user_prompt or "",
                agent_reasoning=session.latest_reasoning or "",
                num_ai_config_corrections=session.num_corrections,
            )
        except Exception:
            # Metadata enrichment is nice to have; the validated JSON is the important part.
            pass

    def _build_config_prompt(self, vendor, user_prompt: str) -> str:
        return f"""
            You are a part of an expert system that helps to facilitate federated learning experiments by generating
            experiment configurations that are then used as an input for experimentation framework.
            The following file contents provide the specification of possible experiment configuration entries,
            and it's validation logic:
            {vendor.framework_config_context}
            Examples of correct configuration:
            {vendor.framework_positive_config_examples}
            Based on this information, generate the experiment configuration that would address the following research question:
            {user_prompt}
        """

    async def start_session(self, user_prompt: str) -> AgentSession:
        session = AgentSession(id=uuid.uuid4().hex, vendor=self._new_vendor())
        session.user_prompt = user_prompt
        session.phase = "generating_config"
        session.pending = True
        self._append_message(session, "user", user_prompt)
        self.sessions[session.id] = session
        self._bump(session)
        asyncio.create_task(self._generate_config(session, self._build_config_prompt(session.vendor, user_prompt)))
        return session

    async def refine_session(self, session_id: str, refinement_request: str) -> AgentSession:
        session = self.get_session(session_id)
        session.phase = "refining_config"
        session.pending = True
        session.error = None
        self._append_message(session, "user", refinement_request)
        self._bump(session)

        prompt = f"""
            The user was not satisfied with the previous configuration.
            Adjust the JSON configuration as follows: {refinement_request}.
            Generate and provide the refined JSON configuration file only.
        """
        asyncio.create_task(self._generate_config(session, prompt))
        return session

    async def _generate_config(self, session: AgentSession, prompt: str) -> None:
        try:
            generated_config, corrections = await asyncio.to_thread(
                session.vendor._generate_and_validate_config,
                prompt,
                "Awaiting AI to generate case study...",
            )
            reasoning = await asyncio.to_thread(session.vendor._explain_config, generated_config)

            session.num_corrections += corrections
            session.latest_config_text = generated_config
            session.latest_config_path = session.vendor.generated_config_json_filepath
            session.latest_reasoning = reasoning
            session.phase = "config_ready"
            session.pending = False
            session.error = None

            self._prepare_final_config_metadata(session)

            self._append_message(
                session,
                "assistant",
                (
                    f"{reasoning}\n\n"
                    "The generated configuration is validated and ready. "
                    "It has been reflected in the JSON editor. Send a correction here if you want changes, "
                    "or click Start to run it."
                ),
            )
        except Exception as exc:
            session.phase = "error"
            session.pending = False
            session.error = str(exc)
            self._append_message(session, "assistant", f"I hit an error while generating the configuration: {exc}")
        finally:
            self._bump(session)

    def get_session(self, session_id: str) -> AgentSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown agent session: {session_id}") from exc

    async def wait_for_update(self, session_id: str, since: int, timeout_seconds: float = 25.0) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session.version <= since:
            try:
                await asyncio.wait_for(session.update_event.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                pass
            finally:
                session.update_event.clear()
        return self._snapshot(session)

    def attach_run(self, session_id: str, run_id: str) -> None:
        session = self.get_session(session_id)
        session.run_id = run_id
        session.phase = "running_experiment"
        session.pending = False
        self._append_message(
            session,
            "assistant",
            "Execution started. I’ll wait for the run to finish and then summarize the observations here.",
        )
        self._bump(session)

    async def analyze_results(
            self,
            session_id: str,
            run_id: str,
            results_dirname: str,
            generated_strategy_config_json_filepath: str,
            simulation_crashed: bool = False,
    ) -> AgentSession:
        session = self.get_session(session_id)
        if session.pending and session.phase == "analyzing_results":
            return session
        if session.result_summary_run_id == run_id:
            return session

        session.phase = "analyzing_results"
        session.pending = True
        session.error = None
        self._bump(session)

        asyncio.create_task(
            self._analyze_results_task(
                session,
                run_id=run_id,
                results_dirname=results_dirname,
                generated_strategy_config_json_filepath=generated_strategy_config_json_filepath,
                simulation_crashed=simulation_crashed,
            )
        )
        return session

    async def _analyze_results_task(
            self,
            session: AgentSession,
            run_id: str,
            results_dirname: str,
            generated_strategy_config_json_filepath: str,
            simulation_crashed: bool,
    ) -> None:
        try:
            summary = await asyncio.to_thread(
                self._reason_about_results,
                session.vendor,
                results_dirname,
                simulation_crashed,
                session.user_prompt,
                generated_strategy_config_json_filepath,
            )
            session.result_summary = summary
            session.result_summary_run_id = run_id
            session.phase = "completed"
            session.pending = False
            session.error = None
            self._append_message(session, "assistant", summary)
        except Exception as exc:
            session.phase = "error"
            session.pending = False
            session.error = str(exc)
            self._append_message(session, "assistant", f"I hit an error while analyzing the results: {exc}")
        finally:
            self._bump(session)

    @staticmethod
    def _reason_about_results(
            vendor,
            results_dirname: str,
            simulation_crashed: bool,
            user_request: str | None,
            generated_strategy_config_json_filepath: str,
    ) -> str:
        results_content = vendor._get_all_result_context(results_dirname)
        prompt = ""

        if user_request is not None:
            prompt += f"""
                You are a part of an expert system that helps to facilitate federated learning experiments by parsing
                quantitative metrics that were collected during FL execution in this expert system. When parsing metrics,
                you must link the user request to observed results and provide recommendations and reasoning on observed
                patterns. User-provided case study objective: {user_request}. The following file contents provide
                the specification of possible experiment configuration entries, and it's validation logic:
                {vendor.framework_config_context}. User requested to execute the following configuration:
                {load_json_as_str(generated_strategy_config_json_filepath)}
            """

        if simulation_crashed:
            prompt += "This experiment crashed due to a problem, but incomplete results are still available."

        prompt += f"""
            Results of the case study include plots and csv files.
            You are provided with the content of the following files:
            - exec_stats_*.csv: stats summary for the entire case execution from "simulation_strategies" JSON.
            - round_metrics_*.csv: metrics for each aggregation round, including model performance and client removal
              performance. Also includes the time used by the aggregation strategy at each round.
            Each * corresponds to the entry from simulation_strategies array, representing the results for separate case.
            Below are file contents:
            {results_content}
            Explain the results, and provide recommendation for the user, considering their prompt and
            case study observations from the resulting files.
        """

        return vendor._get_new_conversation_response(
            content=prompt,
            awaiting_message="Awaiting AI explanation of results...",
        )
