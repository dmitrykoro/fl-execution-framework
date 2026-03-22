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
class AnalysisEvidence:
    title: str
    body: str


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
    evidence: list[AnalysisEvidence] = field(default_factory=list)
    update_event: asyncio.Event = field(default_factory=asyncio.Event)


class DatabaseAnalysisManager:
    def __init__(self, ai_agent_config_filename: str = "ai_agent_config.json") -> None:
        self.ai_agent_config_filename = ai_agent_config_filename
        self.sessions: dict[str, AnalysisSession] = {}

    def _new_vendor(self):
        return AIAgent(ai_agent_config_filename=self.ai_agent_config_filename).ai_vendor_api

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
            "evidence": [{"title": e.title, "body": e.body} for e in session.evidence],
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

    async def start_session(self, question: str) -> AnalysisSession:
        session = AnalysisSession(id=uuid.uuid4().hex, vendor=self._new_vendor())
        session.question = question
        session.phase = "planning_queries"
        session.pending = True
        session.messages.append({"role": "user", "text": question})
        self.sessions[session.id] = session
        self._bump(session)
        asyncio.create_task(self._run_analysis(session))
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

    async def _run_analysis(self, session: AnalysisSession) -> None:
        try:
            query_plan = await asyncio.to_thread(self._generate_query_plan, session.vendor, session.question or "")

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
                session.vendor,
                session.question or "",
                query_results,
            )

            answer = str(analysis_payload.get("answer", "")).strip() or "No answer returned."
            evidence_items = [
                AnalysisEvidence(title=str(item.get("title", "Evidence")), body=str(item.get("body", "")).strip())
                for item in analysis_payload.get("evidence", [])
            ]

            if not evidence_items:
                evidence_items = [
                    AnalysisEvidence(
                        title=result["name"],
                        body=self._format_query_result_markdown(result),
                    )
                    for result in query_results
                ]
            else:
                evidence_items.extend(
                    AnalysisEvidence(
                        title=f"{result['name']} data",
                        body=self._format_query_result_markdown(result),
                    )
                    for result in query_results
                )

            session.messages.append({"role": "assistant", "text": answer})
            session.evidence = evidence_items
            session.phase = "completed"
            session.pending = False
            session.error = None
        except Exception as exc:
            session.phase = "error"
            session.pending = False
            session.error = str(exc)
            session.messages.append(
                {"role": "assistant", "text": f"I hit an error while analyzing the database: {exc}"}
            )
        finally:
            self._bump(session)

    def _generate_query_plan(self, vendor, question: str) -> dict[str, Any]:
        prompt = f"""
            You are a PostgreSQL analytics assistant for a federated-learning experiment database.
            
            Database schema:
            ```python
            {SCHEMA_CONTEXT}
            ```
            
            Task:
            - Read the user question.
            - Produce one or more read-only PostgreSQL queries that answer it.
            - You may only use SELECT statements and CTEs.
            - Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE, COPY, CALL, DO, MERGE.
            - Prefer concise queries that retrieve only useful columns.
            - If date filtering is helpful, use PostgreSQL syntax.
            
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
        raw = vendor._get_new_conversation_response(
            content=prompt,
            awaiting_message="Awaiting AI to plan database queries...",
        )
        return self._parse_json_response(raw)

    async def _execute_remote_sql(self, sql: str) -> dict[str, Any]:
        experiment_server_url = obtain_env_var("EXPERIMENT_SERVER_URL")
        token = obtain_env_var("EXPERIMENT_SERVER_TOKEN")

        async with aiohttp.ClientSession() as client:
            async with client.post(
                    f"{experiment_server_url}/query",
                    json={"sql": sql},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"DB query failed ({resp.status}): {text}")
                return await resp.json()

    def _analyze_results(self, vendor, question: str, query_results: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = f"""
            You are an analytics assistant for federated-learning experiment history.
            The user asked:
            {question}
            
            Below are the executed SQL queries and their results:
            {json.dumps(query_results, indent=2, default=str)}
            
            Return JSON only with this exact shape:
            {{
              "answer": "Concise markdown answer for the user.",
              "evidence": [
                {{
                  "title": "Short evidence title",
                  "body": "Markdown summary with concrete numbers or observations."
                }}
              ]
            }}
            
            Requirements:
            - Answer the user's actual question, not just restate the data.
            - Cite concrete values from the query results.
            - If the data is insufficient, say so plainly.
            - Keep the answer readable and direct.
        """

        raw = vendor._get_new_conversation_response(
            content=prompt,
            awaiting_message="Awaiting AI to explain retrieved database evidence...",
        )
        return self._parse_json_response(raw)

    @staticmethod
    def _format_query_result_markdown(result: dict[str, Any]) -> str:
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        sql = result.get("sql", "")
        purpose = result.get("purpose", "")
        row_count = result.get("row_count", len(rows))

        lines = []
        if purpose:
            lines.append(purpose)
            lines.append("")
        lines.append(f"Rows returned: {row_count}")
        lines.append("")
        lines.append("SQL:")
        lines.append("```sql")
        lines.append(sql)
        lines.append("```")
        lines.append("")

        if not columns:
            lines.append("No tabular result returned.")
            return "\n".join(lines).strip()

        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows[:25]:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")

        if len(rows) > 25:
            lines.append("")
            lines.append(f"... truncated to first 25 rows out of {len(rows)}.")

        return "\n".join(lines).strip()
