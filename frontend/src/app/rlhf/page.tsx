"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Suspense } from "react";

type Dataset = { id: string; name: string; scenario_count: number };
type EvalRun = { id: string; dataset_id: string; dataset_name: string; model: string; metrics: { accuracy: number; correct: number; total: number }; created_at: string };
type Project = { id: string; name: string; total: number; annotated: number; pending: number; preferred_model?: string; rejected_model?: string; strategy?: string };

function RLHFContent() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [evalRuns, setEvalRuns] = useState<EvalRun[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [step, setStep] = useState(1);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [preferredRun, setPreferredRun] = useState("");
  const [rejectedRun, setRejectedRun] = useState("");
  const [strategy, setStrategy] = useState("correct_preferred");
  const [projectName, setProjectName] = useState("");
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    fetch("/api/datasets").then(r => r.json()).then(setDatasets).catch(() => {});
    fetch("/api/eval-runs").then(r => r.json()).then(setEvalRuns).catch(() => {});
    fetch("/api/projects").then(r => r.json()).then(setProjects).catch(() => {});
  }, []);

  useEffect(() => {
    if (!jobId || !running) return;
    const iv = setInterval(async () => {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      if (job.status === "completed" || job.status === "failed") {
        setRunning(false);
        setDone(true);
        fetch("/api/projects").then(r => r.json()).then(setProjects).catch(() => {});
      }
    }, 2000);
    return () => clearInterval(iv);
  }, [jobId, running]);

  const datasetRuns = evalRuns.filter(r => r.dataset_id === selectedDataset);
  const prefRun = evalRuns.find(r => r.id === preferredRun);
  const rejRun = evalRuns.find(r => r.id === rejectedRun);
  const selectedDs = datasets.find(d => d.id === selectedDataset);

  const startRLHF = async () => {
    setRunning(true);
    setDone(false);
    const config: any = {
      name: projectName || "RLHF Pairs", dataset: "mujoco:stacking", job_type: "rlhf",
      dataset_id: selectedDataset, preferred_run_id: preferredRun,
      rejected_run_id: rejectedRun, strategy, num_stable: 0, num_unstable: 0,
    };
    const res = await fetch("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) });
    const job = await res.json();
    setJobId(job.id);
    await fetch(`/api/jobs/${job.id}/run`, { method: "POST" });
  };

  const strategies: { value: string; label: string; desc: string }[] = [
    { value: "correct_preferred", label: "Preferred correct only", desc: "Only include scenarios where the preferred model got it right" },
    { value: "disagreements", label: "Models disagree", desc: "Only where models gave different predictions" },
    { value: "all", label: "All scenarios", desc: "Pair all common scenarios regardless of correctness" },
  ];

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
            <Link href="/evaluate" className="hover:text-gray-900">Evaluate</Link>
            <Link href="/rlhf" className="text-emerald-600 font-medium">RLHF</Link>
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-8 py-10">
        <div className="flex items-center gap-3 mb-8">
          <span className="text-xs font-bold px-2.5 py-1 rounded-full text-white bg-emerald-500">Stage 3</span>
          <h1 className="text-2xl font-semibold text-gray-900">Create RLHF Data</h1>
        </div>

        {/* Stepper */}
        <div className="flex items-center gap-2 mb-8">
          {["Select Dataset", "Choose Models", "Configure & Generate"].map((label, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                step > i + 1 ? "bg-emerald-500 text-white" : step === i + 1 ? "bg-emerald-100 text-emerald-700 border-2 border-emerald-400" : "bg-gray-100 text-gray-400"
              }`}>{step > i + 1 ? "✓" : i + 1}</div>
              <span className={`text-sm ${step === i + 1 ? "font-medium text-gray-900" : "text-gray-400"}`}>{label}</span>
              {i < 2 && <svg className="w-4 h-4 text-gray-300 mx-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>}
            </div>
          ))}
        </div>

        {/* Step 1: Dataset */}
        {step === 1 && (
          <div className="max-w-xl">
            <p className="text-sm text-gray-500 mb-4">Choose the ground truth dataset for pairing</p>
            <div className="space-y-2 mb-6">
              {datasets.map(ds => (
                <button key={ds.id} onClick={() => setSelectedDataset(ds.id)}
                  className={`w-full text-left px-4 py-3 rounded-xl border transition ${
                    selectedDataset === ds.id ? "border-emerald-300 bg-emerald-50" : "border-gray-200 hover:border-gray-300"
                  }`}>
                  <p className="text-sm font-medium text-gray-900">{ds.name}</p>
                  <p className="text-xs text-gray-500">{ds.scenario_count} scenarios</p>
                </button>
              ))}
              {datasets.length === 0 && (
                <div className="text-center py-10 text-gray-400 text-sm">
                  No datasets yet. <Link href="/generate" className="text-emerald-500 underline">Generate one first</Link>
                </div>
              )}
            </div>
            <button onClick={() => setStep(2)} disabled={!selectedDataset}
              className="px-6 py-2.5 rounded-lg bg-emerald-500 text-white text-sm font-medium hover:bg-emerald-600 disabled:opacity-50 transition">
              Continue
            </button>
          </div>
        )}

        {/* Step 2: Models */}
        {step === 2 && (
          <div className="max-w-xl">
            <p className="text-sm text-gray-500 mb-1">
              Dataset: <span className="font-medium text-gray-700">{selectedDs?.name}</span>
            </p>
            {datasetRuns.length < 2 ? (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 mt-4 text-sm text-amber-700">
                <p className="font-medium mb-1">Need at least 2 eval runs</p>
                <p className="text-amber-600">You have {datasetRuns.length} eval run(s) on this dataset. Run more evaluations first.</p>
                <Link href={`/evaluate?datasetId=${selectedDataset}`} className="inline-block mt-3 text-amber-800 underline font-medium">
                  → Go to Evaluate
                </Link>
              </div>
            ) : (
              <>
                <p className="text-sm text-gray-500 mb-4 mt-1">Pick the preferred (teacher) and rejected (student) model</p>
                <div className="space-y-4 mb-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Preferred Model <span className="text-emerald-500">(chosen responses)</span>
                    </label>
                    <div className="space-y-2">
                      {datasetRuns.map(r => (
                        <button key={r.id} onClick={() => setPreferredRun(r.id)}
                          className={`w-full text-left px-4 py-3 rounded-xl border transition ${
                            preferredRun === r.id ? "border-emerald-300 bg-emerald-50" : "border-gray-200 hover:border-gray-300"
                          } ${rejectedRun === r.id ? "opacity-30 cursor-not-allowed" : ""}`}
                          disabled={rejectedRun === r.id}>
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-900">{r.model}</span>
                            <span className="text-sm font-bold text-emerald-600">{r.metrics.accuracy}%</span>
                          </div>
                          <p className="text-xs text-gray-500">{r.metrics.correct}/{r.metrics.total} correct · {new Date(r.created_at).toLocaleDateString()}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Rejected Model <span className="text-rose-500">(rejected responses)</span>
                    </label>
                    <div className="space-y-2">
                      {datasetRuns.filter(r => r.id !== preferredRun).map(r => (
                        <button key={r.id} onClick={() => setRejectedRun(r.id)}
                          className={`w-full text-left px-4 py-3 rounded-xl border transition ${
                            rejectedRun === r.id ? "border-rose-300 bg-rose-50" : "border-gray-200 hover:border-gray-300"
                          }`}>
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-900">{r.model}</span>
                            <span className="text-sm font-bold text-rose-600">{r.metrics.accuracy}%</span>
                          </div>
                          <p className="text-xs text-gray-500">{r.metrics.correct}/{r.metrics.total} correct · {new Date(r.created_at).toLocaleDateString()}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button onClick={() => setStep(1)} className="px-5 py-2.5 rounded-lg border border-gray-200 text-gray-700 text-sm hover:bg-gray-50">Back</button>
                  <button onClick={() => setStep(3)} disabled={!preferredRun || !rejectedRun}
                    className="px-6 py-2.5 rounded-lg bg-emerald-500 text-white text-sm font-medium hover:bg-emerald-600 disabled:opacity-50 transition">
                    Continue
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* Step 3: Configure */}
        {step === 3 && (
          <div className="max-w-xl">
            {/* Summary */}
            <div className="rounded-xl bg-gray-50 p-4 mb-6 text-sm space-y-2">
              <div className="flex justify-between"><span className="text-gray-500">Dataset</span><span className="text-gray-900 font-medium">{selectedDs?.name}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Preferred</span><span className="text-emerald-700 font-medium">{prefRun?.model} ({prefRun?.metrics.accuracy}%)</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Rejected</span><span className="text-rose-700 font-medium">{rejRun?.model} ({rejRun?.metrics.accuracy}%)</span></div>
            </div>

            <div className="space-y-5 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Pairing Strategy</label>
                <div className="space-y-2">
                  {strategies.map(s => (
                    <button key={s.value} onClick={() => setStrategy(s.value)}
                      className={`w-full text-left px-4 py-3 rounded-xl border transition ${
                        strategy === s.value ? "border-emerald-300 bg-emerald-50" : "border-gray-200 hover:border-gray-300"
                      }`}>
                      <p className="text-sm font-medium text-gray-900">{s.label}</p>
                      <p className="text-xs text-gray-500">{s.desc}</p>
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Project Name</label>
                <input value={projectName} onChange={e => setProjectName(e.target.value)}
                  placeholder={`${prefRun?.model || ""} vs ${rejRun?.model || ""}`}
                  className="w-full rounded-lg border border-gray-200 px-3 py-2.5 text-sm focus:border-emerald-400 focus:ring-1 focus:ring-emerald-400 outline-none" />
              </div>
            </div>

            {done && (
              <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 mb-4 text-sm text-emerald-700 flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg>
                RLHF pairs generated successfully! See below.
              </div>
            )}

            <div className="flex gap-3">
              <button onClick={() => setStep(2)} className="px-5 py-2.5 rounded-lg border border-gray-200 text-gray-700 text-sm hover:bg-gray-50">Back</button>
              <button onClick={startRLHF} disabled={running}
                className={`px-6 py-2.5 rounded-lg text-white text-sm font-medium transition-all ${running ? "bg-emerald-400 cursor-wait" : "bg-emerald-500 hover:bg-emerald-600 active:scale-95"} disabled:opacity-50`}>
                {running ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                    Generating...
                  </span>
                ) : "Generate RLHF Pairs"}
              </button>
            </div>
          </div>
        )}

        {/* Projects */}
        {projects.length > 0 && (
          <div className="mt-12 pt-8 border-t border-gray-100">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">RLHF Projects</h2>
            <div className="space-y-3">
              {projects.map(p => {
                const pct = Math.round((p.annotated / Math.max(p.total, 1)) * 100);
                return (
                  <div key={p.id} className="flex items-center gap-5 rounded-xl border border-gray-200 px-5 py-4">
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-900 truncate">{p.name}</p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {p.total} pairs · {p.annotated} annotated
                        {p.preferred_model && ` · ${p.preferred_model} vs ${p.rejected_model}`}
                      </p>
                    </div>
                    <div className="w-20">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-gray-100 rounded-full">
                          <div className="h-1.5 bg-emerald-500 rounded-full" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Link href={`/annotate?projectId=${p.id}&review=true`} className="px-3 py-1.5 text-sm rounded-lg border border-gray-200 text-gray-700 hover:bg-gray-50 transition">Review</Link>
                      {p.pending > 0 && (
                        <Link href={`/annotate?projectId=${p.id}`} className="px-3 py-1.5 text-sm rounded-lg bg-emerald-500 text-white hover:bg-emerald-600 transition">Annotate</Link>
                      )}
                      <a href={`/api/export?projectId=${p.id}&format=dpo&all=true`} className="px-3 py-1.5 text-sm rounded-lg border border-gray-200 text-gray-700 hover:bg-gray-50 transition">Export</a>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function RLHFPage() {
  return <Suspense><RLHFContent /></Suspense>;
}
