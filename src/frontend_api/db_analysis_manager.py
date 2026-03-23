from __future__ import annotations

import asyncio
import json
import time
import uuid

import aiohttp

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.ai_agent_wrapper import AIAgent
from src.utils.common import obtain_env_var

SCHEMA_CONTEXT = Path("src/experiment_database/models.py").read_text(encoding="utf-8")


@dataclass
class AnalysisSession:
    id: str
    vendor: Any
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 0
    phase: str = "idle"
    pending: bool = False
    error: str | None = None
    question: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    evidence_markdown: str = ""
    update_event: asyncio.Event = field(default_factory=asyncio.Event)


class DatabaseAnalysisManager:
    def __init__(self, ai_agent_config_filename: str = "ai_agent_config.json") -> None:
        self.ai_agent_config_filename = ai_agent_config_filename
        self.sessions: dict[str, AnalysisSession] = {}

    def _new_vendor(self):
        return AIAgent(ai_agent_config_filename=self.ai_agent_config_filename).ai_vendor_api

    def _author_name(self) -> str:
        return str(obtain_env_var("FULL_AUTHOR_NAME"))

    def _author_email(self) -> str:
        return str(obtain_env_var("AUTHOR_EMAIL"))

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {obtain_env_var('EXPERIMENT_SERVER_TOKEN')}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _db_base_url(self) -> str:
        return str(obtain_env_var("EXPERIMENT_SERVER_URL")).rstrip("/")

    def _bump(self, session: AnalysisSession) -> None:
        session.version += 1
        session.updated_at = time.time()
        session.update_event.set()

    def _snapshot(self, session: AnalysisSession) -> dict[str, Any]:
        return {
            "session_id": session.id,
            "version": session.version,
            "phase": session.phase,
            "pending": session.pending,
            "error": session.error,
            "question": session.question,
            "messages": session.messages,
            "evidence_markdown": session.evidence_markdown,
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_code_fences(text)
        return json.loads(cleaned)

    async def list_sessions(self) -> list[dict[str, Any]]:
        data = await self._db_request(
            "GET",
            "/analysis_sessions",
            params={"author_email": self._author_email()},
        )
        return list(data.get("items", []))

    async def load_session(self, session_id: str) -> AnalysisSession:
        remote = await self._db_request("GET", f"/analysis_sessions/{session_id}")
        session = self.sessions.get(session_id)
        if session is None:
            session = AnalysisSession(id=session_id, vendor=self._new_vendor())
            self.sessions[session_id] = session

        session.messages = [
            {"role": "assistant" if item.get("role") == "agent" else "user", "text": str(item.get("payload", ""))}
            for item in remote.get("messages", [])
        ]
        session.evidence_markdown = str(remote.get("payload", "") or "")
        session.pending = False
        session.error = None
        session.phase = "completed" if session.messages else "idle"
        self._bump(session)
        return session

    async def start_session(self, question: str, session_id: str | None = None) -> AnalysisSession:
        if session_id:
            session = await self.load_session(session_id)
        else:
            session_id = uuid.uuid4().hex
            await self._db_request(
                "POST",
                "/analysis_sessions",
                json_payload={
                    "id": session_id,
                    "author_name": self._author_name(),
                    "author_email": self._author_email(),
                    "payload": "",
                },
            )
            session = AnalysisSession(id=session_id, vendor=self._new_vendor())
            self.sessions[session.id] = session

        session.question = question
        session.phase = "planning_queries"
        session.pending = True
        session.error = None
        session.messages.append({"role": "user", "text": question})
        await self._append_remote_message(session.id, role="user", payload=question)
        self._bump(session)
        asyncio.create_task(self._run_analysis(session, question))
        return session

    def get_session(self, session_id: str) -> AnalysisSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown analysis session: {session_id}") from exc

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

    async def _run_analysis(self, session: AnalysisSession, question: str) -> None:
        try:
            query_plan = await asyncio.to_thread(self._generate_query_plan, session, question)

            session.phase = "retrieving_data"
            self._bump(session)

            query_results: list[dict[str, Any]] = []
            for item in query_plan.get("queries", []):
                sql = str(item.get("sql", "")).strip()
                if not sql:
                    continue
                result = await self._execute_remote_sql(sql)
                query_results.append(
                    {
                        "name": item.get("name") or "Query result",
                        "purpose": item.get("purpose") or "",
                        "sql": sql,
                        "columns": result.get("columns", []),
                        "rows": result.get("rows", []),
                        "row_count": result.get("row_count", 0),
                    }
                )

            session.phase = "reasoning_about_data"
            self._bump(session)

            analysis_payload = await asyncio.to_thread(
                self._analyze_results,
                session,
                question,
                query_results,
            )

            answer = str(analysis_payload.get("answer", "")).strip() or "No answer returned."
            evidence_markdown = str(analysis_payload.get("evidence_markdown", "")).strip()
            if not evidence_markdown:
                evidence_markdown = self._build_fallback_markdown(query_results)

            if session.evidence_markdown.strip():
                session.evidence_markdown = f"{session.evidence_markdown.rstrip()}\n\n---\n\n{evidence_markdown.lstrip()}".strip()
            else:
                session.evidence_markdown = evidence_markdown

            session.messages.append({"role": "assistant", "text": answer})
            await self._append_remote_message(session.id, role="agent", payload=answer)
            await self._update_remote_payload(session.id, payload=session.evidence_markdown, mode="replace")
            session.phase = "completed"
            session.pending = False
            session.error = None
        except Exception as exc:
            session.phase = "error"
            session.pending = False
            session.error = str(exc)
            error_text = f"I hit an error while analyzing the database: {exc}"
            session.messages.append({"role": "assistant", "text": error_text})
            try:
                await self._append_remote_message(session.id, role="agent", payload=error_text)
            except Exception:
                pass
        finally:
            self._bump(session)

    def _conversation_context(self, session: AnalysisSession) -> str:
        if not session.messages:
            return "No previous conversation."
        return "\n".join(f"{item['role']}: {item['text']}" for item in session.messages[-12:])

    def _generate_query_plan(self, session: AnalysisSession, question: str) -> dict[str, Any]:
        prompt = f"""
            You are a PostgreSQL analytics assistant for a federated-learning experiment database.

            Database schema:
            ```python
            {SCHEMA_CONTEXT}
            ```

            Existing conversation context:
            {self._conversation_context(session)}

            Existing explainability markdown for this session:
            {session.evidence_markdown or "No prior evidence yet."}

            Task:
            - Read the user's newest question.
            - Produce one or more read-only PostgreSQL queries that answer it.
            - You may only use SELECT statements and CTEs.
            - Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, COPY, CALL, DO, MERGE.
            - Prefer concise queries that retrieve only useful columns.
            - If date filtering is helpful, use PostgreSQL syntax.
            - Reuse prior context when the new question is a follow-up.

            Return JSON only with this exact shape:
            {{
              "queries": [
                {{
                  "name": "short label",
                  "purpose": "why this query helps",
                  "sql": "SELECT ..."
                }}
              ]
            }}

            User question:
            {question}
        """
        raw = session.vendor._get_new_conversation_response(
            content=prompt,
            awaiting_message="Awaiting AI to plan database queries...",
        )
        return self._parse_json_response(raw)

    async def _execute_remote_sql(self, sql: str) -> dict[str, Any]:
        return await self._db_request("POST", "/query", json_payload={"sql": sql}, timeout_seconds=30)

    def _analyze_results(self, session: AnalysisSession, question: str, query_results: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = f"""
            You are an analytics assistant for federated-learning experiment history.

            Conversation context:
            {self._conversation_context(session)}

            Existing explainability markdown:
            {session.evidence_markdown or "No prior evidence yet."}

            The user asked:
            {question}

            Below are the executed SQL queries and their results:
            {json.dumps(query_results, indent=2, default=str)}

            Return JSON only with this exact shape:
            {{
              "answer": "Concise markdown answer for the user.",
              "evidence_markdown": "A markdown document for the explainability panel. Include SQL, tables, and concrete evidence."
            }}

            Requirements:
            - The answer should directly answer the user's question.
            - The evidence markdown should be readable as a standalone report section.
            - Include concrete numbers and caveats where useful.
            - If results are sparse, say so clearly.
        """
        raw = session.vendor._get_new_conversation_response(
            content=prompt,
            awaiting_message="Awaiting AI to summarize database findings...",
        )
        return self._parse_json_response(raw)

    def _build_fallback_markdown(self, query_results: list[dict[str, Any]]) -> str:
        if not query_results:
            return "## Retrieved statistics\n\nNo rows were returned for this analysis."
        sections = ["## Retrieved statistics"]
        for result in query_results:
            sections.append(f"### {result['name']}")
            if result.get("purpose"):
                sections.append(result["purpose"])
            sections.append("```sql")
            sections.append(result["sql"])
            sections.append("```")
            sections.append(self._format_query_result_markdown(result))
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _format_query_result_markdown(result: dict[str, Any]) -> str:
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        if not columns:
            return "No columns returned."
        header = "| " + " | ".join(str(column) for column in columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body_rows = []
        for row in rows:
            body_rows.append(
                "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
            )
        if not body_rows:
            body_rows.append("| " + " | ".join("No rows" if idx == 0 else "" for idx, _ in enumerate(columns)) + " |")
        return "\n".join([header, separator, *body_rows])

    async def _append_remote_message(self, session_id: str, role: str, payload: str) -> None:
        await self._db_request(
            "POST",
            f"/analysis_sessions/{session_id}/messages",
            json_payload={"role": role, "payload": payload},
        )

    async def _update_remote_payload(self, session_id: str, payload: str, mode: str = "replace") -> None:
        await self._db_request(
            "PUT",
            f"/analysis_sessions/{session_id}/payload",
            json_payload={"payload": payload, "mode": mode},
        )

    async def _db_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
        timeout_seconds: float = 30,
    ) -> dict[str, Any]:
        async with aiohttp.ClientSession() as client:
            async with client.request(
                method,
                f"{self._db_base_url()}{path}",
                params=params,
                json=json_payload,
                headers=self._auth_headers(),
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"Remote DB request failed ({resp.status}): {text}")
                return await resp.json()
