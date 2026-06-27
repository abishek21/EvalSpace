"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

type Dataset = { id: string; name: string; scenario_count: number };
type EvalRun = {
  id: string; dataset_id: string; dataset_name: string; model: string;
  metrics: {
    accuracy: number; correct: number; total: number;
    task_type?: string; positive_label?: string; negative_label?: string;
    stable_accuracy: number; unstable_accuracy: number;
    stable_correct: number; stable_total: number;
    unstable_correct: number; unstable_total: number;
    positive_accuracy?: number; negative_accuracy?: number;
    positive_correct?: number; positive_total?: number;
    negative_correct?: number; negative_total?: number;
  };
  created_at: string; job_id?: string;
};

const ENV_BADGE: Record<string, { icon: string; label: string }> = {
  stacking_stability: { icon: "🏗️", label: "Stacking" },
  collision_prediction: { icon: "🎱", label: "Collision" },
  spatial_fitting: { icon: "🔲", label: "Spatial Fitting" },
  shelf_fitting: { icon: "🗄️", label: "Shelf Fitting" },
  door_passage: { icon: "🚪", label: "Door Passage" },
  ui_visual: { icon: "🖥️", label: "UI Visual" },
  perspective_taking: { icon: "📷", label: "Perspective Taking" },
  allocentric: { icon: "🧭", label: "Allocentric" },
};

function EvaluateContent() {
  const searchParams = useSearchParams();
  const presetDatasetId = searchParams.get("datasetId") || "";

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [evalRuns, setEvalRuns] = useState<EvalRun[]>([]);
  const [selectedDataset, setSelectedDataset] = useState(presetDatasetId);
  const [model, setModel] = useState("gpt-4o");
  const [azureEndpoint, setAzureEndpoint] = useState("https://akaudiobot.services.ai.azure.com/openai/v1");
  const [azureKey, setAzureKey] = useState("");
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState("");
  const [progress, setProgress] = useState<any>(null);
  const [selectedRun, setSelectedRun] = useState<EvalRun | null>(null);
  const [tab, setTab] = useState<"new" | "history">("new");
  const [promptTemplate, setPromptTemplate] = useState({ system: "", user: "" });
  const [customPromptSuffix, setCustomPromptSuffix] = useState("");

  useEffect(() => {
    fetch("/api/datasets").then(r => r.json()).then(setDatasets).catch(() => {});
    fetch("/api/eval-runs").then(r => r.json()).then(d => { setEvalRuns(d); if (d.length > 0) setTab("history"); }).catch(() => {});
  }, []);

  // Fetch prompt template when dataset changes
  useEffect(() => {
    if (!selectedDataset) { setPromptTemplate({ system: "", user: "" }); return; }
    fetch(`/api/datasets/${selectedDataset}`).then(r => r.json()).then(ds => {
      const tt = ds.task_type || "stacking_stability";
      const templates: Record<string, { system: string; user: string }> = {
        stacking_stability: {
          system: "You are a physics reasoning assistant. Analyze images of stacked objects and predict stability.",
          user: "Look at these images of objects stacked on a table.\n\n{scenario question}\n\nAnswer with PREDICTION: STABLE or UNSTABLE, then REASONING.",
        },
        collision_prediction: {
          system: "You are a physics reasoning assistant. Predict whether a pushed object will collide with a target.",
          user: "Look at these images of objects on a table.\n\n{scenario question}\n\nAnswer with PREDICTION: HIT or MISS, then REASONING.",
        },
        spatial_fitting: {
          system: "You are a spatial reasoning assistant. Determine whether the object could fit through the opening.",
          user: "{scenario question}\n\nAnswer with PREDICTION: FITS or DOES NOT FIT, then REASONING.",
        },
        shelf_fitting: {
          system: "You are a spatial reasoning assistant. Determine whether the object could fit through the opening.",
          user: "{scenario question}\n\nAnswer with PREDICTION: FITS or DOES NOT FIT, then REASONING.",
        },
        door_passage: {
          system: "You are a spatial reasoning assistant. Determine whether the furniture fits through the doorway.",
          user: "{scenario question}\n\nAnswer with PREDICTION: FITS or DOES NOT FIT, then REASONING.",
        },
        perspective_taking: {
          system: "You are a spatial reasoning expert. Look at the image carefully and answer the question.\n\nAnswer with EXACTLY this format:\nPREDICTION: <object name or left/right>\nREASONING: <your reasoning in 1-2 sentences>",
          user: "{scenario question}",
        },
        allocentric: {
          system: "You are a spatial reasoning expert. Look at the image carefully and answer the question.\n\nAnswer with EXACTLY this format:\nPREDICTION: <object name or left/right>\nREASONING: <your reasoning in 1-2 sentences>",
          user: "{scenario question}",
        },
      };
      setPromptTemplate(templates[tt] || templates.stacking_stability);
    }).catch(() => {});
  }, [selectedDataset]);

  useEffect(() => {
    if (!jobId || !running) return;
    const iv = setInterval(async () => {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      setProgress(job.progress);
      if (job.status === "completed" || job.status === "failed") {
        setRunning(false);
        const runs = await fetch("/api/eval-runs").then(r => r.json());
        setEvalRuns(runs);
        setTab("history");
      }
    }, 2000);
    return () => clearInterval(iv);
  }, [jobId, running]);

  const startEval = async () => {
    if (!selectedDataset) return;
    setRunning(true);
    setProgress(null);
    const config: any = {
      name: "Eval " + model, dataset: "mujoco:stacking", job_type: "evaluate",
      dataset_id: selectedDataset, model, num_stable: 0, num_unstable: 0,
    };
    const isLocalGpu = model.startsWith("qwen") && !model.includes("/");
    if (!isLocalGpu) config.azure_config = { endpoint: azureEndpoint, api_key: azureKey };
    const res = await fetch("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) });
    const job = await res.json();
    setJobId(job.id);
    const runRes = await fetch(`/api/jobs/${job.id}/run`, { method: "POST" });
    if (!runRes.ok) {
      const err = await runRes.json().catch(() => ({ detail: "Unknown error" }));
      alert(`Failed to start eval: ${err.detail || runRes.statusText}`);
      setRunning(false);
      return;
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <nav className="border-b border-gray-100 px-8 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-sky-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-sm font-bold">S</span>
            </div>
            <span className="font-semibold text-gray-900">Spatial Reasoning Lab</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-gray-500">
            <Link href="/" className="hover:text-gray-900">Home</Link>
            <Link href="/datasets" className="hover:text-gray-900">Datasets</Link>
            <Link href="/generate" className="hover:text-gray-900">Generate</Link>
            <Link href="/evaluate" className="text-violet-600 font-medium">Evaluate</Link>
            <Link href="/rlhf" className="hover:text-gray-900">RLHF</Link>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="flex items-center gap-3 mb-8">
          <span className="text-xs font-bold px-2.5 py-1 rounded-full text-white bg-violet-500">Stage 2</span>
          <h1 className="text-2xl font-semibold text-gray-900">Evaluate Model</h1>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 bg-gray-100 rounded-lg p-1 w-fit">
          <button onClick={() => { setTab("new"); setSelectedRun(null); }}
            className={`px-4 py-2 text-sm rounded-md transition ${tab === "new" ? "bg-white shadow-sm font-medium text-gray-900" : "text-gray-500 hover:text-gray-700"}`}>
            + New Evaluation
          </button>
          <button onClick={() => setTab("history")}
            className={`px-4 py-2 text-sm rounded-md transition ${tab === "history" ? "bg-white shadow-sm font-medium text-gray-900" : "text-gray-500 hover:text-gray-700"}`}>
            Eval Runs ({evalRuns.length})
          </button>
        </div>

        {tab === "new" && (
          <div className="max-w-3xl">
            <p className="text-sm text-gray-500 mb-6">Run a VLM against a ground truth dataset and measure accuracy</p>
            <div className="rounded-xl border border-gray-200 p-6">
              <div className="grid grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Dataset</label>
                  <select value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}
                    className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm">
                    <option value="">Select a dataset...</option>
                    {datasets.map(ds => <option key={ds.id} value={ds.id}>{ds.name} ({ds.scenario_count} scenarios)</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Model</label>
                  <select value={model} onChange={e => {
                    setModel(e.target.value);
                    const isOpenRouter = e.target.value.includes("/") || e.target.value.startsWith("grok");
                    if (isOpenRouter) setAzureEndpoint("https://openrouter.ai/api/v1");
                    else if (!azureEndpoint.includes("openrouter")) setAzureEndpoint("https://akaudiobot.services.ai.azure.com/openai/v1");
                  }}
                    className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm">
                    <option value="gpt-4o">GPT-4o (Azure OpenAI)</option>
                    <option value="gpt-4.1">GPT-4.1 (Azure OpenAI)</option>
                    <option value="gpt-5">GPT-5 (Azure OpenAI)</option>
                    <option value="Phi-4">Phi-4 (Azure AI)</option>
                    <option value="x-ai/grok-4.3">Grok 4.3 (OpenRouter)</option>
                    <option value="anthropic/claude-opus-4.8">Claude Opus 4.8 (OpenRouter)</option>
                    <option value="anthropic/claude-sonnet-4">Claude Sonnet 4 (OpenRouter)</option>
                    <option value="google/gemini-2.5-pro">Gemini 2.5 Pro (OpenRouter)</option>
                    <option value="google/gemini-3.5-flash">Gemini 3.5 Flash (OpenRouter)</option>
                    <option value="openai/gpt-5.5">GPT-5.5 (OpenRouter)</option>
                    <option value="deepseek/deepseek-v4-pro">DeepSeek V4 Pro (OpenRouter)</option>
                    <option value="qwen/qwen-2.5-vl-72b-instruct">Qwen 2.5 VL 72B (OpenRouter)</option>
                    <option value="qwen/qwen-2.5-vl-7b-instruct">Qwen 2.5 VL 7B (OpenRouter)</option>
                    <option value="qwen2.5-vl-3b">Qwen 2.5-VL 3B (GPU)</option>
                  </select>
                </div>
              </div>

              {!(model.startsWith("qwen") && !model.includes("/")) && (
                <div className="grid grid-cols-2 gap-6 mb-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Azure Endpoint</label>
                    <input value={azureEndpoint} onChange={e => setAzureEndpoint(e.target.value)}
                      className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">API Key</label>
                    <input type="password" value={azureKey} onChange={e => setAzureKey(e.target.value)}
                      placeholder="Enter Azure API key" className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" />
                  </div>
                </div>
              )}

              {/* Prompt Template Preview */}
              {selectedDataset && promptTemplate.system && (
                <div className="mb-6 rounded-lg border border-gray-100 bg-gray-50 p-4">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">📋 Prompt Template</p>
                  <div className="mb-3">
                    <p className="text-xs font-medium text-gray-600 mb-1">System Prompt</p>
                    <pre className="text-xs bg-white border border-gray-200 rounded-md p-2.5 whitespace-pre-wrap text-gray-700">{promptTemplate.system}</pre>
                  </div>
                  <div className="mb-3">
                    <p className="text-xs font-medium text-gray-600 mb-1">User Prompt</p>
                    <pre className="text-xs bg-white border border-gray-200 rounded-md p-2.5 whitespace-pre-wrap text-gray-700">{promptTemplate.user}</pre>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-gray-600 mb-1">Custom Suffix <span className="text-gray-400">(optional — appended to user prompt)</span></p>
                    <textarea
                      value={customPromptSuffix}
                      onChange={e => setCustomPromptSuffix(e.target.value)}
                      placeholder="e.g., Think step by step before answering."
                      className="w-full rounded-md border border-gray-200 px-2.5 py-2 text-xs font-mono resize-y min-h-[40px]"
                      rows={2}
                    />
                  </div>
                </div>
              )}

              <button onClick={startEval} disabled={running || !selectedDataset || (!(model.startsWith("qwen") && !model.includes("/")) && !azureKey.trim())}
                className={`px-6 py-2.5 rounded-lg text-white text-sm font-medium transition-all ${running ? "bg-violet-400 cursor-wait" : "bg-violet-500 hover:bg-violet-600 active:scale-95"} disabled:opacity-50`}>
                {running ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                    Running...
                  </span>
                ) : "Run Evaluation"}
              </button>

              {running && progress && (
                <div className="mt-4">
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-gray-100 rounded-full">
                      <div className="h-2 bg-violet-500 rounded-full transition-all" style={{ width: `${((progress.scenes_processed || 0) / Math.max(progress.scenarios_total || 1, 1)) * 100}%` }} />
                    </div>
                    <span className="text-sm text-gray-500 tabular-nums">{progress.scenes_processed || 0}/{progress.scenarios_total || "?"} · {progress.correct || 0} correct</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {tab === "history" && (
          <div className="grid grid-cols-[300px_1fr] gap-6 min-h-[400px]">
            {/* Left: run list */}
            <div className="space-y-2">
              {evalRuns.length === 0 ? (
                <p className="text-sm text-gray-400 py-8 text-center">No evaluations yet</p>
              ) : evalRuns.map(run => (
                <div key={run.id} className={`relative group w-full text-left px-4 py-3 rounded-xl border transition ${
                    selectedRun?.id === run.id ? "border-violet-300 bg-violet-50" : "border-gray-200 hover:border-gray-300"
                  }`}>
                  <button onClick={() => setSelectedRun(run)} className="w-full text-left">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-gray-900">{run.model}</span>
                      <span className="text-sm font-bold text-violet-600 pr-5">{run.metrics.accuracy}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {run.metrics.task_type && ENV_BADGE[run.metrics.task_type] && (
                        <span className="text-xs px-1.5 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-medium">
                          {ENV_BADGE[run.metrics.task_type].icon} {ENV_BADGE[run.metrics.task_type].label}
                        </span>
                      )}
                      <span className="text-xs text-gray-500">{run.dataset_name} · {new Date(run.created_at).toLocaleDateString()}</span>
                    </div>
                  </button>
                  <button onClick={async (e) => { e.stopPropagation(); if (!confirm("Delete this eval run?")) return; await fetch(`/api/eval-runs/${run.id}`, { method: "DELETE" }); setEvalRuns(prev => prev.filter(r => r.id !== run.id)); if (selectedRun?.id === run.id) setSelectedRun(null); }}
                    className="absolute top-2.5 right-2.5 p-1 rounded-md text-gray-300 hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-all"
                    title="Delete eval run">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                  </button>
                </div>
              ))}
            </div>

            {/* Right: detail */}
            <div className="rounded-xl border border-gray-200 overflow-hidden">
              {!selectedRun ? (
                <div className="flex items-center justify-center h-full text-sm text-gray-400">Select an evaluation to see details</div>
              ) : (
                <div className="p-6 space-y-6">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h2 className="text-xl font-semibold text-gray-900">{selectedRun.model}</h2>
                        {selectedRun.metrics.task_type && ENV_BADGE[selectedRun.metrics.task_type] && (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-medium">
                            {ENV_BADGE[selectedRun.metrics.task_type].icon} {ENV_BADGE[selectedRun.metrics.task_type].label}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-500 mt-1">
                        Dataset: {selectedRun.dataset_name} · {new Date(selectedRun.created_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-3xl font-bold text-violet-600">{selectedRun.metrics.accuracy}%</p>
                      <p className="text-sm text-gray-500">{selectedRun.metrics.correct}/{selectedRun.metrics.total} correct</p>
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-emerald-50 rounded-xl p-4 border border-emerald-100">
                      <p className="text-xs text-emerald-600 font-medium mb-1">{selectedRun.metrics.positive_label ? selectedRun.metrics.positive_label.charAt(0).toUpperCase() + selectedRun.metrics.positive_label.slice(1) : "Stable"} Scenarios</p>
                      <p className="text-2xl font-bold text-emerald-700">{selectedRun.metrics.positive_accuracy ?? selectedRun.metrics.stable_accuracy}%</p>
                      <p className="text-xs text-emerald-600 mt-1">{selectedRun.metrics.positive_correct ?? selectedRun.metrics.stable_correct}/{selectedRun.metrics.positive_total ?? selectedRun.metrics.stable_total} correct</p>
                    </div>
                    <div className="bg-rose-50 rounded-xl p-4 border border-rose-100">
                      <p className="text-xs text-rose-600 font-medium mb-1">{selectedRun.metrics.negative_label ? selectedRun.metrics.negative_label.charAt(0).toUpperCase() + selectedRun.metrics.negative_label.slice(1) : "Unstable"} Scenarios</p>
                      <p className="text-2xl font-bold text-rose-700">{selectedRun.metrics.negative_accuracy ?? selectedRun.metrics.unstable_accuracy}%</p>
                      <p className="text-xs text-rose-600 mt-1">{selectedRun.metrics.negative_correct ?? selectedRun.metrics.unstable_correct}/{selectedRun.metrics.negative_total ?? selectedRun.metrics.unstable_total} correct</p>
                    </div>
                  </div>

                  {/* Accuracy bar */}
                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Overall Accuracy</p>
                    <div className="h-3 bg-gray-100 rounded-full">
                      <div className="h-3 bg-violet-500 rounded-full transition-all" style={{ width: `${selectedRun.metrics.accuracy}%` }} />
                    </div>
                  </div>

                  {/* Details */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-xs text-gray-400 mb-1">Answer Model</p>
                      <p className="font-medium text-gray-900">{selectedRun.model}</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-xs text-gray-400 mb-1">Questions</p>
                      <p className="font-medium text-gray-900">Template-based</p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-3 pt-4 border-t border-gray-100">
                    <Link href={`/annotate?projectId=${selectedRun.id}&review=true&type=eval`}
                      className="px-4 py-2 text-sm rounded-lg border border-gray-200 text-gray-700 hover:bg-gray-50 transition">
                      Review Results
                    </Link>
                    <Link href={`/annotate?projectId=${selectedRun.id}&type=eval`}
                      className="px-4 py-2 text-sm rounded-lg bg-violet-500 text-white hover:bg-violet-600 transition">
                      Annotate
                    </Link>
                    <a href={`/api/export?projectId=${selectedRun.id}&format=full&all=true`}
                      className="px-4 py-2 text-sm rounded-lg border border-gray-200 text-gray-700 hover:bg-gray-50 transition">
                      Export JSON
                    </a>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function EvaluatePage() {
  return <Suspense><EvaluateContent /></Suspense>;
}
