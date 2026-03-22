import {useEffect, useMemo, useRef, useState} from "react";

/**
 * InteFL Research Framework – UI (API-backed)
 * Tabs: Run | Analyze | Docs | AI Settings | Config templates
 *
 * Key UX requirement: logs stay visible at the bottom regardless of tab.
 *
 * Backend API (via Vite proxy recommended):
 *
 * Required (implemented in this UI):
 *  - GET  /api/health
 *  - POST /api/validate   { config: {...} } -> { ok: boolean, errors: string[] }
 *  - POST /api/start      { config: {...}, log_level?: "INFO"|... } -> { run_id: string }
 *  - GET  /api/logs?run_id=... -> { lines: string[] }
 *  - GET  /api/status?run_id=... -> { status: "RUNNING"|"EXITED"|"UNKNOWN", exit_code?: number }
 *  - GET  /api/experiments -> { items: Experiment[] }
 *
 * Optional (UI will use if available; falls back gracefully if not):
 *  - POST /api/agent/suggest  { text: string } -> { config: {...} }
 *  - POST /api/analyze/chat   { question: string } -> { answer: string, evidence?: {title:string, body:string}[] }
 *  - GET  /api/docs           -> { markdown: string }
 *  - PUT  /api/docs           { markdown: string } -> { ok: boolean }
 *  - GET  /api/templates      -> { items: Template[] }
 *  - POST /api/templates      { name: string, config: {...} } -> { item: Template }
 */

// --------------------
// Helpers
// --------------------

type JsonValue = any;

type JsonObject = Record<string, JsonValue>;

type ApiValidateResponse = { ok: boolean; errors?: string[] };

type ApiStartResponse = { run_id: string };

type ApiLogsResponse = { lines: string[] };

type ApiStatusResponse = { run_id: string; status: "RUNNING" | "EXITED" | "UNKNOWN"; exit_code?: number };

type ExperimentLink = { label: string; href: string };

type Experiment = {
    id: string;
    title: string;
    status: string;
    createdAt: string;
    config: JsonObject;
    links: ExperimentLink[];
    raw?: JsonObject;
};

type BackendExperiment = {
    id: number | string;
    created_at: string | null;
    is_successful: boolean | null;
    total_exec_time: number | null;
    error_message: string | null;
    full_author_name: string | null;
    author_email: string | null;
    config: {
        strategy: string | null;
        dataset: string | null;
        num_rounds: number | null;
        num_clients: number | null;
        num_malicious_clients: number | null;
        json: JsonObject | string | null;
    } | null;
    hardware: {
        cpu: string | null;
        arch: string | null;
        ram: string | null;
        gpu_name: string | null;
        gpu_ram: string | null;
        is_executed_on: string | null;
    } | null;
    clients_at_round: Array<{
        round_number: number | null;
        client_number: number | null;
        is_aggregated: boolean | null;
        attack_type: string | null;
    }>;
    metric_entries: Array<{
        round_number: number | null;
        metric_name: string | null;
        metric_type: string | null;
        value: number | null;
        client_number: number | null;
    }>;
};

type ApiExperimentsResponse = { count?: number; experiments?: BackendExperiment[]; items?: Experiment[] };

type ChatMessage = { role: "user" | "assistant"; text: string };

type Template = { id: string; name: string; config: JsonObject };

type EvidenceItem = { title: string; body: string };

type DocsSection = {
    key: string;
    description: string;
    options: { value: string; description: string }[];
};

function cx(...xs: Array<string | false | null | undefined>) {
    return xs.filter(Boolean).join(" ");
}

function prettyJson(objOrStr: unknown) {
    try {
        const obj = typeof objOrStr === "string" ? JSON.parse(objOrStr) : objOrStr;
        return JSON.stringify(obj, null, 2);
    } catch {
        return typeof objOrStr === "string" ? objOrStr : "";
    }
}

function validateJson(text: string) {
    const errors: string[] = [];
    try {
        const parsed = JSON.parse(text);
        if (!parsed || typeof parsed !== "object") errors.push("JSON must be an object");
    } catch {
        errors.push("Invalid JSON syntax");
    }
    return errors;
}

function badge(status: string) {
    const base = "text-[11px] rounded-full px-2 py-0.5";
    if (status === "SUCCESS") return cx(base, "bg-emerald-50 text-emerald-700");
    if (status === "RUNNING") return cx(base, "bg-amber-50 text-amber-800");
    if (status === "FAILED") return cx(base, "bg-rose-50 text-rose-700");
    return cx(base, "bg-zinc-100 text-zinc-700");
}

function formatCreatedAt(value: string | null | undefined) {
    if (!value) return "Unknown time";
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
}

function parseMaybeJson(value: unknown): JsonObject | null {
    if (!value) return null;
    if (typeof value === "object" && !Array.isArray(value)) return value as JsonObject;
    if (typeof value === "string") {
        try {
            const parsed = JSON.parse(value);
            if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
        } catch {
            return null;
        }
    }
    return null;
}

function summarizeMetrics(metricEntries: BackendExperiment["metric_entries"]) {
    const latestByName = new Map<string, { round_number: number | null; value: number | null }>();
    for (const m of metricEntries ?? []) {
        if (!m.metric_name) continue;
        const prev = latestByName.get(m.metric_name);
        if (!prev || (m.round_number ?? -1) >= (prev.round_number ?? -1)) {
            latestByName.set(m.metric_name, {round_number: m.round_number, value: m.value});
        }
    }
    return latestByName;
}

function mapBackendExperiment(exp: BackendExperiment): Experiment {
    const cfg = exp.config ?? null;
    const parsedJson = parseMaybeJson(cfg?.json ?? null);
    const mergedConfig: JsonObject = {
        ...(parsedJson ?? {}),
        strategy: cfg?.strategy ?? parsedJson?.strategy ?? null,
        dataset: cfg?.dataset ?? parsedJson?.dataset ?? null,
        num_rounds: cfg?.num_rounds ?? parsedJson?.num_rounds ?? null,
        num_clients: cfg?.num_clients ?? parsedJson?.num_clients ?? null,
        num_malicious_clients: cfg?.num_malicious_clients ?? parsedJson?.num_malicious_clients ?? null,
    };

    const strategy = cfg?.strategy ?? "Unknown strategy";
    const dataset = cfg?.dataset ?? "Unknown dataset";
    const status = exp.is_successful === true ? "SUCCESS" : exp.is_successful === false ? "FAILED" : "UNKNOWN";

    return {
        id: String(exp.id),
        title: `${strategy}, ${dataset}`,
        status,
        createdAt: formatCreatedAt(exp.created_at),
        config: mergedConfig,
        links: [
            {label: "Logs", href: "#"},
            {label: "Config JSON", href: "#"},
            {label: "Metrics", href: "#"},
        ],
        raw: exp as unknown as JsonObject,
    };
}

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(path, {
        ...init,
        headers: {
            "Content-Type": "application/json",
            ...(init?.headers ?? {}),
        },
    });

    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`API ${res.status} ${res.statusText}${text ? `: ${text}` : ""}`);
    }

    return (await res.json()) as T;
}

function safeParseJson(text: string): { ok: true; value: JsonObject } | { ok: false; error: string } {
    try {
        const v = JSON.parse(text);
        if (!v || typeof v !== "object" || Array.isArray(v)) {
            return {ok: false, error: "JSON must be an object"};
        }
        return {ok: true, value: v};
    } catch {
        return {ok: false, error: "Invalid JSON syntax"};
    }
}

/**
 * Parse a constrained markdown format into structured documentation sections.
 * Expected input pattern:
 * - **`key`**
 *   Optional description...
 *   - `value`: description
 */
function parseDocsMarkdown(md: string): DocsSection[] {
    const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");

    const sections: DocsSection[] = [];
    const sectionHeaderRe = /^\s*-\s*\*\*`([^`]+)`\*\*\s*$/;
    const optionRe = /^\s*-\s*`([^`]+)`\s*:\s*(.*)\s*$/;

    let i = 0;
    while (i < lines.length) {
        const header = lines[i].match(sectionHeaderRe);
        if (!header) {
            i += 1;
            continue;
        }

        const key = header[1].trim();
        i += 1;

        const descParts: string[] = [];
        const options: { value: string; description: string }[] = [];

        while (i < lines.length && !sectionHeaderRe.test(lines[i])) {
            const line = lines[i];
            const opt = line.match(optionRe);
            if (opt) {
                options.push({value: opt[1].trim(), description: (opt[2] || "").trim()});
            } else {
                const cleaned = line.replace(/^\s+/, "").trimEnd();
                if (cleaned && !cleaned.startsWith("-") && cleaned !== "Options:") {
                    descParts.push(cleaned);
                }
            }
            i += 1;
        }

        sections.push({key, description: descParts.join("\n").trim(), options});
    }

    return sections;
}

// Map docs keys to config fields.
// NOTE: You previously asked whether dataset_keyword should map to `dataset`.
// Current behavior: dataset_keyword -> dataset.
function docsKeyToConfigField(key: string) {
    const map: Record<string, string> = {
        dataset_keyword: "dataset",
        bad_clients: "bad clients",
        bad_clients_keyword: "bad clients",
    };
    return map[key] ?? key;
}

function coerceDocValue(value: string) {
    const v = String(value).trim();
    if (/^\d+$/.test(v)) return Number(v);
    if (/^\[.*\]$/.test(v)) {
        try {
            return JSON.parse(v);
        } catch {
            return v;
        }
    }
    return v;
}

function loadTemplatesFromStorage(): Template[] {
    try {
        const raw = localStorage.getItem("intefl.templates");
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed;
    } catch {
        return [];
    }
}

function saveTemplatesToStorage(templates: Template[]) {
    try {
        localStorage.setItem("intefl.templates", JSON.stringify(templates));
    } catch {
        // ignore
    }
}

// --------------------
// Defaults
// --------------------

const DEFAULT_JSON: JsonObject = {
    dataset: "femnist",
    "bad clients": "20%",
};

function suggestConfigFromText(text: string): JsonObject {
    const lower = text.toLowerCase();

    let dataset = "femnist";
    if (lower.includes("medical") || lower.includes("pneumonia")) {
        dataset = "pneumoniamnist";
    } else if (lower.includes("traffic") || lower.includes("sign")) {
        dataset = "its";
    } else if (lower.includes("iid")) {
        dataset = "femnist_iid";
    } else if (lower.includes("non-iid") || lower.includes("niid")) {
        dataset = "femnist_niid";
    }

    let strategy = "PID";
    if (lower.includes("multikrum")) strategy = "MultiKrum";
    if (lower.includes("trimmed")) strategy = "TrimmedMean";

    const percentMatch = lower.match(/(\d+)\s*%/);
    const badClients = percentMatch ? `${percentMatch[1]}%` : DEFAULT_JSON["bad clients"];

    return {
        ...DEFAULT_JSON,
        dataset,
        strategy,
        "bad clients": badClients,
    };
}

const DEFAULT_DOCS_MD = `# Configuration reference

- **\`dataset_keyword\`**  
  Dataset used for execution. Options:
  - \`femnist_iid\`: handwritten digit subset (0-9), 10 classes, IID distribution, 100 clients.
  - \`femnist_niid\`: same, but the data is distributed in non-iid manner, according to authors' description. 16 clients max.
  - \`its\`: intelligent Transportation Systems domain, binary classification (traffic sign vs stop sign), 12 clients.
  - \`pneumoniamnist\`: medical imaging (pneumonia diagnosis), binary classification.

- **\`bad_clients\`**  
  Malicious clients definition. Options:
  - \`20%\`: treat 20% of clients as malicious.
  - \`10%\`: treat 10% of clients as malicious.
  - \`[1,3]\`: explicit malicious client IDs.
  - \`10\`: explicit count of malicious clients.

- **\`strategy\`**  
  Aggregation strategy. Options:
  - \`PID\`: PID-based robust aggregation.
  - \`MultiKrum\`: Multi-Krum.
  - \`TrimmedMean\`: Trimmed Mean.
`;

// --------------------
// UI components
// --------------------

function ShellCard({title, subtitle, right, children}: any) {
    return (
        <div className="rounded-2xl bg-white shadow-sm ring-1 ring-zinc-200">
            <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
                <div>
                    <div className="text-sm font-semibold">{title}</div>
                    {subtitle ? <div className="text-xs text-zinc-500">{subtitle}</div> : null}
                </div>
                {right ? right : null}
            </div>
            {children}
        </div>
    );
}

function Tabs({active, setActive}: any) {
    const tabs = ["Run", "Analyze", "Docs", "AI Settings", "Config templates"];
    return (
        <div className="mt-6 flex w-full items-center gap-2 border-b border-zinc-200">
            {tabs.map((t) => {
                const on = t === active;
                return (
                    <button
                        key={t}
                        onClick={() => setActive(t)}
                        className={cx(
                            "relative -mb-px px-4 py-2 text-sm font-medium",
                            on ? "text-zinc-900" : "text-zinc-500 hover:text-zinc-800"
                        )}
                    >
                        {t}
                        {on && <span className="absolute left-0 right-0 -bottom-px h-0.5 bg-zinc-900"/>}
                    </button>
                );
            })}
        </div>
    );
}

function ChatPanel({title, subtitle, messages, draft, setDraft, onSend, placeholder, footerHint}: any) {
    return (
        <ShellCard title={title} subtitle={subtitle}>
            <div className="h-72 overflow-auto px-4 py-3">
                {messages.map((m: ChatMessage, idx: number) => (
                    <div
                        key={idx}
                        className={cx("mb-3 flex", m.role === "user" ? "justify-end" : "justify-start")}
                    >
                        <div
                            className={cx(
                                "max-w-[85%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-sm",
                                m.role === "user" ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-900"
                            )}
                        >
                            {m.text}
                        </div>
                    </div>
                ))}
            </div>
            <div className="border-t border-zinc-100 px-4 py-3">
                <div className="flex items-center gap-2">
                    <input
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") onSend();
                        }}
                        placeholder={placeholder}
                        className="w-full rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-zinc-300"
                    />
                    <button
                        onClick={onSend}
                        className="rounded-xl bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-800"
                    >
                        Send
                    </button>
                </div>
                {footerHint ? <div className="mt-2 text-xs text-zinc-500">{footerHint}</div> : null}
            </div>
        </ShellCard>
    );
}

function GlobalLogs({
                        lines,
                        runId,
                        status,
                        onClear,
                    }: {
    lines: string[];
    runId: string | null;
    status: string | null;
    onClear: () => void;
}) {
    const [hidden, setHidden] = useState(false);

    // panel height (px)
    const [height, setHeight] = useState<number>(220); // default expanded-ish
    const heightRef = useRef(height);
    useEffect(() => {
        heightRef.current = height;
    }, [height]);

    // resize dragging
    const draggingRef = useRef(false);
    const startYRef = useRef(0);
    const startHeightRef = useRef(0);

    const beginResize = (e: React.MouseEvent) => {
        draggingRef.current = true;
        startYRef.current = e.clientY;
        startHeightRef.current = heightRef.current;

        // prevent selecting text while dragging
        document.body.style.userSelect = "none";
    };

    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!draggingRef.current) return;
            const dy = startYRef.current - e.clientY; // dragging up increases height
            const next = Math.min(640, Math.max(120, startHeightRef.current + dy));
            setHeight(next);
        };

        const onUp = () => {
            if (!draggingRef.current) return;
            draggingRef.current = false;
            document.body.style.userSelect = "";
        };

        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
    }, []);

    // autoscroll logic
    const scrollRef = useRef<HTMLDivElement | null>(null);
    const stickToBottomRef = useRef(true);

    const onScroll = () => {
        const el = scrollRef.current;
        if (!el) return;

        // if user is within 24px from bottom -> keep auto-scrolling
        const distanceFromBottom = el.scrollHeight - (el.scrollTop + el.clientHeight);
        stickToBottomRef.current = distanceFromBottom < 24;
    };

    useEffect(() => {
        if (hidden) return;
        const el = scrollRef.current;
        if (!el) return;
        if (!stickToBottomRef.current) return;

        el.scrollTop = el.scrollHeight;
    }, [lines, hidden, height]);

    // When un-hiding, jump to bottom
    useEffect(() => {
        if (!hidden) {
            const el = scrollRef.current;
            if (el) el.scrollTop = el.scrollHeight;
            stickToBottomRef.current = true;
        }
    }, [hidden]);

    // Hidden state: show a tiny bar to bring it back
    if (hidden) {
        return (
            <div className="fixed bottom-0 left-0 right-0 z-50 bg-white shadow-lg ring-1 ring-zinc-200">
                <div className="mx-auto max-w-6xl px-4">
                    <div className="flex items-center justify-between px-4 py-2">
                        <div className="flex items-center gap-2">
                            <div className="text-sm font-semibold">Execution logs</div>
                            {runId ? (
                                <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-700">
                  {runId}
                </span>
                            ) : null}
                            {status ? (
                                <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-700">
                  {status}
                </span>
                            ) : null}
                        </div>
                        <button
                            onClick={() => setHidden(false)}
                            className="rounded-xl bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-zinc-800"
                        >
                            Show
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed bottom-0 left-0 right-0 z-50 bg-white shadow-lg ring-1 ring-zinc-200">
            <div className="mx-auto max-w-6xl px-4">
                {/* drag handle */}
                <div
                    onMouseDown={beginResize}
                    className="cursor-row-resize px-4 pt-2"
                    title="Drag to resize"
                >
                    <div className="mx-auto h-1.5 w-12 rounded-full bg-zinc-200"/>
                </div>

                <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
                    <div>
                        <div className="flex items-center gap-2">
                            <div className="text-sm font-semibold">Execution logs</div>
                            {runId ? (
                                <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-700">
                  {runId}
                </span>
                            ) : null}
                            {status ? (
                                <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-700">
                  {status}
                </span>
                            ) : null}
                        </div>
                        <div className="text-xs text-zinc-500">Streaming output from the run</div>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setHeight((h) => Math.min(640, h + 140))}
                            className="rounded-xl border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium hover:bg-zinc-50"
                            title="Quick expand"
                        >
                            Expand
                        </button>
                        <button
                            onClick={onClear}
                            className="rounded-xl border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium hover:bg-zinc-50"
                        >
                            Clear
                        </button>
                        <button
                            onClick={() => setHidden(true)}
                            className="rounded-xl bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-zinc-800"
                        >
                            Hide
                        </button>
                    </div>
                </div>

                <div
                    ref={scrollRef}
                    onScroll={onScroll}
                    style={{maxHeight: height}}
                    className="overflow-auto px-4 py-3"
                >
          <pre
              className="whitespace-pre-wrap rounded-xl bg-zinc-50 p-3 text-xs leading-5 text-zinc-800 ring-1 ring-zinc-100">
            {lines.join("\n")}
          </pre>
                </div>
            </div>
        </div>
    );
}

// --------------------
// Minimal dev tests (runs only in dev)
// --------------------

function runDevTests() {
    // prettyJson formats object
    const pj = prettyJson({a: 1});
    if (!pj.includes('"a": 1')) throw new Error("prettyJson test failed");

    // validateJson catches invalid JSON
    const errs1 = validateJson("{");
    if (errs1.length === 0) throw new Error("validateJson invalid syntax test failed");

    // validateJson catches missing fields
    const errs2 = validateJson("{}");
    if (!errs2.some((e) => e.includes("dataset"))) {
        throw new Error("validateJson missing dataset test failed");
    }

    // validateJson passes minimal valid config
    const errs3 = validateJson('{"dataset":"femnist","bad clients":"20%"}');
    if (errs3.length !== 0) throw new Error("validateJson valid config test failed");

    // parseDocsMarkdown extracts sections and options
    const md = "- **`dataset_keyword`**\n  Options:\n  - `a`: aaa\n  - `b`: bbb\n\n- **`bad_clients`**\n  Options:\n  - `20%`: twenty";
    const sections = parseDocsMarkdown(md);
    if (sections.length !== 2) throw new Error("parseDocsMarkdown section count test failed");
    if (sections[0].key !== "dataset_keyword") throw new Error("parseDocsMarkdown key test failed");
    if (sections[0].options.length !== 2) throw new Error("parseDocsMarkdown options count test failed");
    if (sections[1].options[0].value !== "20%") throw new Error("parseDocsMarkdown option value test failed");

    // logs join safety
    const joined = ["a", "b"].join("\n");
    if (joined !== "a\nb") throw new Error("logs join test failed");
}

if (typeof import.meta !== "undefined" && (import.meta as any).env && (import.meta as any).env.DEV) {
    try {
        runDevTests();
    } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
    }
}

// --------------------
// App
// --------------------

export default function App() {
    const [activeTab, setActiveTab] = useState("Run");

    // LOGS (global): UI events + current run output
    const [uiLogs, setUiLogs] = useState<string[]>(["[ui] Ready."]);
    const [runLogs, setRunLogs] = useState<string[]>([]);
    const [runId, setRunId] = useState<string | null>(null);
    const [runStatus, setRunStatus] = useState<string | null>(null);

    const appendUiLog = (line: string) => setUiLogs((prev) => [...prev, line]);

    const mergedLogs = useMemo(() => {
        const all = [...uiLogs];
        if (runId) all.push(`[ui] Attached to run_id=${runId}`);
        return [...all, ...runLogs];
    }, [uiLogs, runLogs, runId]);

    // Poll logs/status for active run
    const pollTimer = useRef<number | null>(null);
    const statusTimer = useRef<number | null>(null);

    useEffect(() => {
        // cleanup any prior polling
        if (pollTimer.current) window.clearInterval(pollTimer.current);
        if (statusTimer.current) window.clearInterval(statusTimer.current);

        if (!runId) {
            setRunLogs([]);
            setRunStatus(null);
            return;
        }

        // logs poll
        pollTimer.current = window.setInterval(async () => {
            try {
                const data = await apiJson<ApiLogsResponse>(`/api/logs?run_id=${encodeURIComponent(runId)}`);
                setRunLogs(data.lines ?? []);
            } catch (e: any) {
                appendUiLog(`[ui] logs poll error: ${e?.message || e}`);
            }
        }, 1000);

        // status poll
        statusTimer.current = window.setInterval(async () => {
            try {
                const s = await apiJson<ApiStatusResponse>(`/api/status?run_id=${encodeURIComponent(runId)}`);
                setRunStatus(s.status);
            } catch {
                // status is optional; ignore if not implemented
            }
        }, 2000);

        return () => {
            if (pollTimer.current) window.clearInterval(pollTimer.current);
            if (statusTimer.current) window.clearInterval(statusTimer.current);
        };
    }, [runId]);

    // RUN: agentic chat (still mocked client-side)
    const [runChatMessages, setRunChatMessages] = useState<ChatMessage[]>([
        {role: "assistant", text: "Describe your use-case in natural language. I will propose a JSON config."},
    ]);
    const [runChatDraft, setRunChatDraft] = useState("");

    // RUN: JSON
    const [jsonText, setJsonText] = useState(prettyJson(DEFAULT_JSON));
    const [jsonStatus, setJsonStatus] = useState<{ ok: boolean; message: string }>(
        {ok: true, message: "Not validated yet"}
    );
    const [apiBusy, setApiBusy] = useState(false);

    // RUN: history (API-backed)
    const [experiments, setExperiments] = useState<Experiment[]>([]);
    const [experimentsLoading, setExperimentsLoading] = useState(false);
    const [selectedExpId, setSelectedExpId] = useState<string | null>(null);

    const selectedExperiment = useMemo(() => {
        if (!selectedExpId) return null;
        return experiments.find((e) => e.id === selectedExpId) ?? null;
    }, [experiments, selectedExpId]);

    const refreshExperiments = async () => {
        setExperimentsLoading(true);
        try {
            const data = await apiJson<ApiExperimentsResponse>("/api/get_experiments");
            const mapped = Array.isArray(data.experiments)
                ? data.experiments.map(mapBackendExperiment)
                : Array.isArray(data.items)
                    ? data.items
                    : [];

            setExperiments(mapped);
            if (!selectedExpId && mapped.length > 0) {
                setSelectedExpId(mapped[0].id);
            }
            appendUiLog(`[ui] loaded ${mapped.length} experiments from API`);
        } catch (e: any) {
            appendUiLog(`[ui] experiments fetch error: ${e?.message || e}`);
        } finally {
            setExperimentsLoading(false);
        }
    };

    // ANALYZE
    const [anChatMessages, setAnChatMessages] = useState<ChatMessage[]>([
        {role: "assistant", text: "Ask about experiment outcomes and I will summarize the retrieved evidence."},
    ]);
    const [anChatDraft, setAnChatDraft] = useState("");
    const [evidence, setEvidence] = useState<EvidenceItem[]>([
        {
            title: "No evidence loaded yet",
            body: "Run an analysis query to populate this panel with retrieved statistics.",
        },
    ]);

    // DOCS
    const [docsMd, setDocsMd] = useState(DEFAULT_DOCS_MD);
    const [docsConfig, setDocsConfig] = useState<JsonObject>(DEFAULT_JSON);
    const docsSections = useMemo(() => parseDocsMarkdown(docsMd), [docsMd]);
    const docsJsonText = useMemo(() => prettyJson(docsConfig), [docsConfig]);

    const applyDocsPick = (key: string, value: string) => {
        const field = docsKeyToConfigField(key);
        const coerced = coerceDocValue(value);

        setDocsConfig((prev) => {
            const next = {...prev, [field]: coerced};
            appendUiLog(`[ui] docs selected ${field}=${JSON.stringify(coerced)}`);
            return next;
        });
    };

    const resetDocsConfig = () => {
        setDocsConfig(DEFAULT_JSON);
        appendUiLog("[ui] docs picks reset");
    };

    // TEMPLATES
    const [templates, setTemplates] = useState<Template[]>(() => loadTemplatesFromStorage());

    useEffect(() => {
        saveTemplatesToStorage(templates);
    }, [templates]);

    useEffect(() => {
        refreshExperiments();
    }, []);

    const onSendRunChat = () => {
        const text = runChatDraft.trim();
        if (!text) return;
        setRunChatMessages((m) => [...m, {role: "user", text}]);
        setRunChatDraft("");

        const suggested = suggestConfigFromText(text);

        setTimeout(() => {
            setRunChatMessages((m) => [
                ...m,
                {
                    role: "assistant",
                    text: "Requested use-case is reflected in generated JSON:\n\n" + prettyJson(suggested),
                },
            ]);
            setJsonText(prettyJson(suggested));
            setJsonStatus({ok: true, message: "Not validated yet"});
            appendUiLog(`[ui] agent suggested config for dataset=${suggested.dataset}`);
        }, 200);
    };

    const onValidate = async () => {
        // client-side quick check first
        const localErrs = validateJson(jsonText);
        if (localErrs.length) {
            setJsonStatus({ok: false, message: localErrs.join(" • ")});
            appendUiLog(`[ui] validate (local) FAILED: ${localErrs.join(" | ")}`);
            return;
        }

        const parsed = safeParseJson(jsonText);
        if (!parsed.ok) {
            const message = "error" in parsed ? parsed.error : "Invalid JSON";
            setJsonStatus({ok: false, message});
            appendUiLog(`[ui] validate parse FAILED: ${message}`);
            return;
        }

        setApiBusy(true);
        try {
            const resp = await apiJson<ApiValidateResponse>("/api/validate_json_config", {
                method: "POST",
                body: JSON.stringify({config: parsed.value}),
            });

            const errs = resp.errors ?? [];
            if (!resp.ok || errs.length) {
                setJsonStatus({ok: false, message: errs.join(" • ") || "Validation failed"});
                appendUiLog(`[ui] validate (api) FAILED: ${errs.join(" | ")}`);
            } else {
                setJsonStatus({ok: true, message: "Validation successful"});
                appendUiLog("[ui] validate (api) OK");
            }
        } catch (e: any) {
            setJsonStatus({ok: false, message: e?.message || String(e)});
            appendUiLog(`[ui] validate (api) error: ${e?.message || e}`);
        } finally {
            setApiBusy(false);
        }
    };

    const onStart = async () => {
        if (!jsonStatus.ok) {
            appendUiLog("[ui] start blocked: fix validation errors first");
            return;
        }

        const parsed = safeParseJson(jsonText);
        if (!parsed.ok) {
            const message = "error" in parsed ? parsed.error : "Invalid JSON";
            setJsonStatus({ok: false, message});
            appendUiLog(`[ui] start blocked: ${message}`);
            return;
        }

        setApiBusy(true);
        try {
            const resp = await apiJson<ApiStartResponse>("/api/start_simulation_runner", {
                method: "POST",
                body: JSON.stringify({config: parsed.value, log_level: "INFO"}),
            });

            setRunId(resp.run_id);
            setRunStatus("RUNNING");
            appendUiLog(`[ui] started run: ${resp.run_id}`);

            // refresh experiments soon after start
            setTimeout(() => {
                refreshExperiments();
            }, 800);

        } catch (e: any) {
            appendUiLog(`[ui] start error: ${e?.message || e}`);
        } finally {
            setApiBusy(false);
        }
    };

    const onSaveTemplate = () => {
        const parsed = safeParseJson(jsonText);
        if (!parsed.ok) {
            const message = "error" in parsed ? parsed.error : "Invalid JSON";
            setJsonStatus({ok: false, message});
            appendUiLog(`[ui] template save blocked: ${message}`);
            return;
        }

        const errs = validateJson(jsonText);
        if (errs.length) {
            setJsonStatus({ok: false, message: errs.join(" • ")});
            appendUiLog(`[ui] template save blocked: ${errs.join(" | ")}`);
            return;
        }

        const id = `tpl_${Date.now()}`;
        const bad = parsed.value["bad clients"];
        const name = `${parsed.value.dataset} • ${Array.isArray(bad) ? bad.join(",") : bad}`;

        setTemplates((t) => [{id, name, config: parsed.value}, ...t]);
        appendUiLog(`[ui] template saved: ${name}`);
        setActiveTab("Config templates");
    };

    const restoreConfig = (config: JsonObject, sourceLabel: string) => {
        setJsonText(prettyJson(config));
        setJsonStatus({ok: true, message: "Not validated yet"});
        appendUiLog(`[ui] restored config from ${sourceLabel}`);
        setActiveTab("Run");
    };

    const onSendAnalyzeChat = () => {
        const text = anChatDraft.trim();
        if (!text) return;
        setAnChatMessages((m) => [...m, {role: "user", text}]);
        setAnChatDraft("");

        setTimeout(() => {
            const answer =
                text.toLowerCase().includes("best") && text.toLowerCase().includes("medical")
                    ? "(mock) According to experimental database, it's PID."
                    : "(mock) I can answer using the experiment database once connected.";

            setAnChatMessages((m) => [...m, {role: "assistant", text: answer}]);

            const newEvidence = {
                title: "Retrieved statistics (mock)",
                body:
                    "• Query inferred from your question\n" +
                    "• Example: top-1 aggregation by avg accuracy on medical datasets\n" +
                    "• Evidence can include tables/plots/links once wired to backend",
            };

            setEvidence((ev) => [newEvidence, ...ev]);
            appendUiLog(`[ui] analyze processed: \"${text}\"`);
        }, 250);
    };

    const clearAllLogs = () => {
        setUiLogs(["[ui] Ready."]);
        setRunLogs([]);
    };

    return (
        <div className="min-h-screen bg-zinc-50 text-zinc-900">
            {/* Content wrapper: add bottom padding so fixed logs never overlap content */}
            <div className="mx-auto max-w-6xl px-4 pt-8 pb-60">
                <div className="flex items-start justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-semibold tracking-tight">InteFL Research Framework</h1>
                        <p className="mt-1 text-sm text-zinc-600">
                            API-backed prototype: execution + analysis + docs (.md → selectable entries) + AI settings +
                            templates.
                        </p>
                    </div>
                    <div className="hidden sm:flex items-center gap-2">
            <span className="rounded-full bg-white px-3 py-1 text-xs text-zinc-600 shadow-sm ring-1 ring-zinc-200">
              v0.4-alpha (Database)
            </span>
                    </div>
                </div>

                <Tabs active={activeTab} setActive={setActiveTab}/>

                {/* RUN */}
                {activeTab === "Run" && (
                    <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
                        <ChatPanel
                            title="AI-assisted execution and analysis"
                            subtitle="User describes use-case in natural language"
                            messages={runChatMessages}
                            draft={runChatDraft}
                            setDraft={setRunChatDraft}
                            onSend={onSendRunChat}
                            placeholder='e.g., "Use femnist with 20% bad clients"'
                            footerHint="Requested use-case is reflected in generated JSON."
                        />

                        <ShellCard
                            title="Manual configuration: JSON"
                            subtitle="User can manually adjust the JSON or configure it from scratch"
                            right={
                                <div
                                    className={cx(
                                        "text-[11px] rounded-full px-2 py-0.5",
                                        jsonStatus.ok ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"
                                    )}
                                    title={jsonStatus.message}
                                >
                                    {apiBusy ? "Working…" : jsonStatus.ok ? "OK" : "Errors"}
                                </div>
                            }
                        >
                            <div className="px-4 py-3">
                                <div className="rounded-xl border border-zinc-200 bg-zinc-50">
                  <textarea
                      value={jsonText}
                      onChange={(e) => setJsonText(e.target.value)}
                      className="h-64 w-full resize-none rounded-xl bg-transparent p-3 font-mono text-[12px] leading-5 outline-none"
                  />
                                </div>

                                <div className="mt-3 grid grid-cols-2 gap-2">
                                    <button
                                        onClick={onValidate}
                                        disabled={apiBusy}
                                        className={cx(
                                            "rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium",
                                            apiBusy ? "opacity-60 cursor-not-allowed" : "hover:bg-zinc-50"
                                        )}
                                    >
                                        Validate
                                    </button>
                                    <button
                                        onClick={onStart}
                                        disabled={!jsonStatus.ok || apiBusy}
                                        className={cx(
                                            "rounded-xl px-3 py-2 text-sm font-medium",
                                            jsonStatus.ok && !apiBusy
                                                ? "bg-zinc-900 text-white hover:bg-zinc-800"
                                                : "bg-zinc-200 text-zinc-500 cursor-not-allowed"
                                        )}
                                        title={jsonStatus.ok ? "Start experiment" : "Validate first"}
                                    >
                                        Start ▷
                                    </button>
                                </div>

                                <button
                                    onClick={onSaveTemplate}
                                    disabled={apiBusy}
                                    className={cx(
                                        "mt-2 w-full rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium",
                                        apiBusy ? "opacity-60 cursor-not-allowed" : "hover:bg-zinc-50"
                                    )}
                                    title="Save this configuration for later reuse"
                                >
                                    Save as a template
                                </button>

                                <div className="mt-2 text-xs text-zinc-500">{jsonStatus.message}</div>
                            </div>
                        </ShellCard>

                        <ShellCard
                            title="Experiments history"
                            subtitle="Experiment history is retrieved from DB and includes links to artifacts"
                            right={
                                <button
                                    onClick={refreshExperiments}
                                    className={cx(
                                        "rounded-xl border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium",
                                        experimentsLoading ? "opacity-60 cursor-not-allowed" : "hover:bg-zinc-50"
                                    )}
                                    disabled={experimentsLoading}
                                >
                                    {experimentsLoading ? "Refreshing…" : "Refresh"}
                                </button>
                            }
                        >
                            <div className="px-4 py-3">
                                <div className="space-y-2">
                                    {experiments.length === 0 ? (
                                        <div
                                            className="rounded-xl border border-dashed border-zinc-300 bg-zinc-50 p-4 text-sm text-zinc-600">
                                            No experiments returned yet.
                                        </div>
                                    ) : (
                                        experiments.map((exp) => {
                                            const active = exp.id === selectedExpId;
                                            return (
                                                <button
                                                    key={exp.id}
                                                    onClick={() => {
                                                        setSelectedExpId(exp.id);
                                                        appendUiLog(`[ui] history selected ${exp.id}`);
                                                    }}
                                                    className={cx(
                                                        "w-full rounded-xl border px-3 py-2 text-left",
                                                        active ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 bg-white hover:bg-zinc-50"
                                                    )}
                                                >
                                                    <div className="flex items-center justify-between gap-2">
                                                        <div className="text-sm font-medium">{exp.title}</div>
                                                        <span className={badge(exp.status)}>{exp.status}</span>
                                                    </div>
                                                    <div className="mt-0.5 text-xs text-zinc-500">{exp.createdAt}</div>
                                                </button>
                                            );
                                        })
                                    )}
                                </div>

                                {selectedExperiment && (
                                    <div className="mt-3 rounded-xl border border-zinc-200 bg-white p-3">
                                        {(() => {
                                            const raw = selectedExperiment.raw as BackendExperiment | undefined;
                                            const latestMetrics = raw ? summarizeMetrics(raw.metric_entries ?? []) : new Map<string, {
                                                round_number: number | null;
                                                value: number | null
                                            }>();
                                            const aggregatedCount = raw ? (raw.clients_at_round ?? []).filter((c) => c.is_aggregated).length : 0;
                                            return (
                                                <>
                                                    <div className="text-xs font-semibold text-zinc-700">Experiment
                                                        details
                                                    </div>
                                                    <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-zinc-600">
                                                        <div><span
                                                            className="font-medium text-zinc-800">Author:</span> {raw?.full_author_name || "Unknown"}
                                                        </div>
                                                        <div><span
                                                            className="font-medium text-zinc-800">Email:</span> {raw?.author_email || "Unknown"}
                                                        </div>
                                                        <div><span
                                                            className="font-medium text-zinc-800">Rounds:</span> {raw?.config?.num_rounds ?? "—"}
                                                        </div>
                                                        <div><span
                                                            className="font-medium text-zinc-800">Clients:</span> {raw?.config?.num_clients ?? "—"}
                                                        </div>
                                                        <div><span
                                                            className="font-medium text-zinc-800">Malicious:</span> {raw?.config?.num_malicious_clients ?? "—"}
                                                        </div>
                                                        <div><span
                                                            className="font-medium text-zinc-800">Aggregated rows:</span> {aggregatedCount}
                                                        </div>
                                                        <div><span
                                                            className="font-medium text-zinc-800">Exec time:</span> {raw?.total_exec_time ?? "—"}
                                                        </div>
                                                        <div><span
                                                            className="font-medium text-zinc-800">GPU:</span> {raw?.hardware?.gpu_name || raw?.hardware?.is_executed_on || "—"}
                                                        </div>
                                                    </div>

                                                    {raw?.error_message ? (
                                                        <div
                                                            className="mt-2 rounded-xl bg-rose-50 px-3 py-2 text-xs text-rose-700 ring-1 ring-rose-100">
                                                            {raw.error_message}
                                                        </div>
                                                    ) : null}

                                                    <div className="mt-3 text-xs font-semibold text-zinc-700">Latest
                                                        metrics
                                                    </div>
                                                    <div className="mt-2 flex flex-wrap gap-2">
                                                        {latestMetrics.size === 0 ? (
                                                            <span
                                                                className="text-xs text-zinc-500">No metrics returned.</span>
                                                        ) : (
                                                            Array.from(latestMetrics.entries()).slice(0, 8).map(([name, metric]) => (
                                                                <span
                                                                    key={name}
                                                                    className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-700"
                                                                    title={`round ${metric.round_number ?? "?"}`}
                                                                >
                                                                        {name}: {metric.value ?? "—"}
                                                                    </span>
                                                            ))
                                                        )}
                                                    </div>

                                                    <div className="mt-3 text-xs font-semibold text-zinc-700">Visualization
                                                        and artifacts links
                                                    </div>
                                                    <div className="mt-2 flex flex-wrap gap-2">
                                                        {(selectedExperiment.links ?? []).map((l) => (
                                                            <a
                                                                key={l.label}
                                                                href={l.href}
                                                                onClick={(e) => {
                                                                    e.preventDefault();
                                                                    appendUiLog(`[ui] link clicked: ${l.label} (${selectedExperiment.id})`);
                                                                }}
                                                                className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-700 hover:bg-zinc-100"
                                                            >
                                                                {l.label}
                                                            </a>
                                                        ))}
                                                    </div>

                                                    <div
                                                        className="mt-2 text-[11px] text-zinc-500">ID: {selectedExperiment.id}</div>
                                                </>
                                            );
                                        })()}
                                    </div>
                                )}
                            </div>
                        </ShellCard>
                    </div>
                )}

                {/* ANALYZE */}
                {activeTab === "Analyze" && (
                    <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
                        <ChatPanel
                            title="AI-assisted statistical analysis"
                            subtitle="(mock) Agent infers SQL request and presents findings"
                            messages={anChatMessages}
                            draft={anChatDraft}
                            setDraft={setAnChatDraft}
                            onSend={onSendAnalyzeChat}
                            placeholder='e.g., "What is the best-performing aggregation on medical images?"'
                            footerHint="Retrieved knowledge is displayed as evidence (explainability)."
                        />

                        <ShellCard title="AI explainability: retrieved statistics"
                                   subtitle="Possibly: plots, tables, numbers (mock)">
                            <div className="px-4 py-3">
                                <div className="space-y-3">
                                    {evidence.map((e: any, i: number) => (
                                        <div key={i} className="rounded-xl border border-zinc-200 bg-white p-3">
                                            <div className="text-xs font-semibold text-zinc-800">{e.title}</div>
                                            <div className="mt-1 whitespace-pre-wrap text-xs text-zinc-600">{e.body}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </ShellCard>
                    </div>
                )}

                {/* DOCS */}
                {activeTab === "Docs" && (
                    <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
                        <ShellCard title="Documentation" subtitle="Loaded from .md and parsed into selectable entries">
                            <div className="px-4 py-4">
                                <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4">
                                    <div className="flex items-center justify-between gap-2">
                                        <div>
                                            <div className="text-sm font-semibold">Markdown source</div>
                                            <div className="text-xs text-zinc-500">Edit .md → entries update below</div>
                                        </div>
                                        <button
                                            onClick={() => {
                                                setDocsMd(DEFAULT_DOCS_MD);
                                                appendUiLog("[ui] docs markdown reset");
                                            }}
                                            className="rounded-xl border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium hover:bg-zinc-50"
                                        >
                                            Reset .md
                                        </button>
                                    </div>

                                    <div className="mt-3 rounded-xl border border-zinc-200 bg-white">
                        <textarea
                            value={docsMd}
                            onChange={(e) => setDocsMd(e.target.value)}
                            className="h-44 w-full resize-none rounded-xl bg-transparent p-3 font-mono text-[12px] leading-5 outline-none"
                        />
                                    </div>

                                    <div className="mt-4">
                                        <div className="text-xs font-semibold text-zinc-800">Parsed entries</div>
                                        <div className="mt-2 space-y-3">
                                            {docsSections.length === 0 ? (
                                                <div
                                                    className="rounded-xl border border-dashed border-zinc-300 bg-white p-4 text-sm text-zinc-600">
                                                    No sections detected. Ensure the pattern: - **`key`** and options: -
                                                    `value`: desc
                                                </div>
                                            ) : (
                                                docsSections.map((sec) => (
                                                    <div key={sec.key}
                                                         className="rounded-xl border border-zinc-200 bg-white p-3">
                                                        <div className="flex items-start justify-between gap-2">
                                                            <div>
                                                                <div
                                                                    className="text-xs font-semibold text-zinc-800">`{sec.key}`
                                                                </div>
                                                                {sec.description ? (
                                                                    <div
                                                                        className="mt-1 whitespace-pre-wrap text-xs text-zinc-600">{sec.description}</div>
                                                                ) : null}
                                                            </div>
                                                            <div
                                                                className="text-[11px] rounded-full bg-zinc-100 px-2 py-0.5 text-zinc-700">
                                                                maps to: {docsKeyToConfigField(sec.key)}
                                                            </div>
                                                        </div>

                                                        <div className="mt-2 flex flex-wrap gap-2">
                                                            {sec.options.length === 0 ? (
                                                                <span
                                                                    className="text-xs text-zinc-500">No options found.</span>
                                                            ) : (
                                                                sec.options.map((opt) => (
                                                                    <button
                                                                        key={`${sec.key}:${opt.value}`}
                                                                        onClick={() => applyDocsPick(sec.key, opt.value)}
                                                                        className="rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-700 hover:bg-zinc-100"
                                                                        title={opt.description || "Click to select"}
                                                                    >
                                                                        {opt.value}
                                                                    </button>
                                                                ))
                                                            )}
                                                        </div>

                                                        {sec.options.some((o) => o.description) ? (
                                                            <div className="mt-2 text-[11px] text-zinc-500">Tip: hover an
                                                                option to see its description.</div>
                                                        ) : null}
                                                    </div>
                                                ))
                                            )}
                                        </div>
                                    </div>

                                    <div className="mt-4 flex items-center justify-between gap-2">
                                        <button
                                            onClick={resetDocsConfig}
                                            className="rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm font-medium hover:bg-zinc-50"
                                        >
                                            Reset picks
                                        </button>
                                        <div className="text-xs text-zinc-500">Click options → updates JSON on the right
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </ShellCard>

                        <ShellCard title="JSON configurator" subtitle="Selections compile here, then send to Run">
                            <div className="px-4 py-4">
                                <div className="rounded-xl border border-zinc-200 bg-zinc-50">
                      <textarea
                          value={docsJsonText}
                          readOnly
                          className="h-72 w-full resize-none rounded-xl bg-transparent p-3 font-mono text-[12px] leading-5 outline-none"
                      />
                                </div>

                                <button
                                    onClick={() => restoreConfig(docsConfig, "docs")}
                                    className="mt-3 w-full rounded-xl bg-emerald-600 px-3 py-2 text-sm font-semibold text-white hover:bg-emerald-700"
                                    title="Send to execution (copies this JSON into Run tab)"
                                >
                                    Send to execution
                                </button>

                                <div className="mt-2 text-xs text-zinc-500">
                                    This copies JSON into <span className="font-medium">Run</span> → Manual configuration.
                                </div>
                            </div>
                        </ShellCard>
                    </div>
                )}

                {/* AI SETTINGS */}
                {activeTab === "AI Settings" && (
                    <div className="mt-6">
                        <ShellCard title="AI Settings"
                                   subtitle="Define the scope of the AI agent (allowed actions, context, constraints)">
                            <div className="px-4 py-4">
                                <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4">
                                    <div className="text-sm font-semibold">Agent scope (mock)</div>
                                    <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
                                        <div className="rounded-xl border border-zinc-200 bg-white p-3">
                                            <div className="text-xs font-semibold text-zinc-800">Allowed</div>
                                            <ul className="mt-2 list-disc pl-5 text-xs text-zinc-600">
                                                <li>Generate candidate JSON configs</li>
                                                <li>Suggest experiment queries for analysis</li>
                                                <li>Explain evidence from retrieved stats</li>
                                            </ul>
                                        </div>
                                        <div className="rounded-xl border border-zinc-200 bg-white p-3">
                                            <div className="text-xs font-semibold text-zinc-800">Not allowed</div>
                                            <ul className="mt-2 list-disc pl-5 text-xs text-zinc-600">
                                                <li>Run experiments without user confirmation</li>
                                                <li>Modify DB records directly</li>
                                                <li>Access external resources without explicit enablement</li>
                                            </ul>
                                        </div>
                                    </div>

                                    <div className="mt-3 rounded-xl border border-zinc-200 bg-white p-3">
                                        <div className="text-xs font-semibold text-zinc-800">Context</div>
                                        <div className="mt-1 text-xs text-zinc-600">
                                            Provide project-specific constraints (datasets).
                                            In a later iteration, this tab can call backend endpoints to persist agent
                                            policies.
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </ShellCard>
                    </div>
                )}

                {/* CONFIG TEMPLATES */}
                {activeTab === "Config templates" && (
                    <div className="mt-6">
                        <ShellCard title="Saved Templates"
                                   subtitle="User can browse saved configs and restore them for execution">
                            <div className="px-4 py-4">
                                {templates.length === 0 ? (
                                    <div
                                        className="rounded-2xl border border-dashed border-zinc-300 bg-zinc-50 p-6 text-sm text-zinc-600">
                                        No templates yet. Go to Run → “Save as a template”.
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                        {templates.map((t) => (
                                            <div key={t.id}
                                                 className="rounded-2xl border border-zinc-200 bg-white p-3 shadow-sm">
                                                <div className="text-sm font-semibold">{t.name}</div>
                                                <pre
                                                    className="mt-2 max-h-40 overflow-auto rounded-xl bg-zinc-50 p-3 text-[11px] leading-5 text-zinc-800 ring-1 ring-zinc-100">
                              {prettyJson(t.config)}
                            </pre>
                                                <button
                                                    onClick={() => restoreConfig(t.config, `template (${t.name})`)}
                                                    className="mt-3 w-full rounded-xl bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-800"
                                                >
                                                    Restore
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </ShellCard>
                    </div>
                )}

                <div className="py-10 text-center text-xs text-zinc-400">UI prototype • API-backed run/logs/history</div>
            </div>

            {/* GLOBAL LOGS (always visible, fixed) */}
            <GlobalLogs lines={mergedLogs} runId={runId} status={runStatus} onClear={clearAllLogs}/>
        </div>
    );
}
