"use client";

import { useState, useEffect, use } from "react";
import Link from "next/link";

type Job = {
  id: string;
  projectId?: string;
  status: string;
  config: {
    dataset: string;
    numScenes: number;
    questionsPerScene: number;
    categories: string[];
    model: string;
    maxViews: number;
    imageResolution: number;
    num_scenes?: number;
    questions_per_scene?: number;
    max_views?: number;
    image_resolution?: number;
    name?: string;
  };
  progress: {
    scenesProcessed: number;
    questionsGenerated: number;
    pairsGenerated: number;
    scenes_processed?: number;
    questions_generated?: number;
    pairs_generated?: number;
  };
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  dataset_id?: string;
};

const STATUS_CONFIG: Record<string, { label: string; color: string }> = {
  queued: { label: "Queued", color: "text-amber-600" },
  downloading: { label: "Downloading", color: "text-sky-600" },
  generating: { label: "Generating", color: "text-violet-600" },
  completed: { label: "Completed", color: "text-emerald-600" },
  failed: { label: "Failed", color: "text-rose-600" },
};

export default function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [job, setJob] = useState<Job | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      const res = await fetch(`/api/jobs/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      if (!cancelled) setJob(data);

      // Keep polling if not terminal
      if (!cancelled && !["completed", "failed"].includes(data.status)) {
        setTimeout(poll, 2000);
      }
    }

    poll();
    return () => { cancelled = true; };
  }, [id]);

  if (!job) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <p className="text-gray-400 text-sm">Loading job...</p>
      </div>
    );
  }

  const status = STATUS_CONFIG[job.status] || STATUS_CONFIG.queued;
  const prog = {
    scenes: job.progress.scenesProcessed ?? job.progress.scenes_processed ?? 0,
    questions: job.progress.questionsGenerated ?? job.progress.questions_generated ?? 0,
    pairs: job.progress.pairsGenerated ?? job.progress.pairs_generated ?? 0,
    scenarios_total: (job.progress as any).scenarios_total ?? 0,
  };
  const numScenes = job.config.numScenes ?? job.config.num_scenes ?? 0;
  const qPerScene = job.config.questionsPerScene ?? job.config.questions_per_scene ?? 1;
  const maxViews = job.config.maxViews ?? job.config.max_views ?? 0;
  const jobType = (job.config as any).job_type || (job.config as any).jobType || "";
  const environment = (job.config as any).environment || "stacking_stability";
  const isGT = jobType === "generate_gt";
  const isEval = jobType === "evaluate";
  const isRLHF = jobType === "rlhf";

  // For GT jobs, progress is scenes_processed / num_scenes
  const totalExpected = isGT ? numScenes : numScenes * qPerScene;
  const currentProgress = isGT ? prog.scenes : prog.pairs;
  const progressPct = totalExpected > 0
    ? Math.min(100, Math.round((currentProgress / totalExpected) * 100))
    : 0;
  const datasetName = job.config.name || `Job ${job.id.slice(0, 8)}`;

  const envLabel = environment === "collision_prediction" ? "Collision Prediction" : "Stacking Stability";
  const envIcon = environment === "collision_prediction" ? "🎱" : "📦";

  return (
    <div className="min-h-screen bg-white">
      <nav className="border-b border-gray-100 px-8 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-sky-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-sm font-bold">S</span>
            </div>
            <Link href="/" className="font-semibold text-gray-900">Spatial Reasoning Lab</Link>
          </div>
          <span className="text-sm text-gray-400 font-mono">{datasetName}</span>
        </div>
      </nav>

      <div className="max-w-3xl mx-auto px-8 py-10">
        {/* Status */}
        <div className="rounded-xl border border-gray-200 p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-2.5 h-2.5 rounded-full ${
              job.status === "generating" ? "bg-violet-500 animate-pulse" :
              job.status === "completed" ? "bg-emerald-500" :
              job.status === "failed" ? "bg-rose-500" : "bg-amber-500"
            }`} />
            <p className={`text-lg font-medium ${status.color}`}>{status.label}</p>
          </div>

          {(job.status === "generating" || job.status === "downloading") && (
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-500 mb-1.5">
                <span>
                  {isGT ? `Simulating scenarios ${envIcon}` :
                   isEval ? "Evaluating with VLM" :
                   isRLHF ? "Pairing responses" : "Processing"}
                </span>
                <span>{currentProgress}/{totalExpected || "?"} · {progressPct}%</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-1.5">
                <div className="bg-sky-500 h-1.5 rounded-full transition-all duration-500" style={{ width: `${Math.max(progressPct, 2)}%` }} />
              </div>
            </div>
          )}

          {isGT ? (
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-2xl font-semibold text-gray-900">{prog.scenes}/{numScenes}</p>
                <p className="text-xs text-gray-500">Scenarios Simulated</p>
              </div>
              <div>
                <p className="text-2xl font-semibold text-gray-900">{prog.pairs}</p>
                <p className="text-xs text-gray-500">Ground Truths Saved</p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-semibold text-gray-900">{prog.scenes}/{numScenes}</p>
                <p className="text-xs text-gray-500">Scenes</p>
              </div>
              <div>
                <p className="text-2xl font-semibold text-gray-900">{prog.questions}</p>
                <p className="text-xs text-gray-500">Questions</p>
              </div>
              <div>
                <p className="text-2xl font-semibold text-gray-900">{prog.pairs}</p>
                <p className="text-xs text-gray-500">Pairs</p>
              </div>
            </div>
          )}

          {job.error && (
            <div className="mt-4 p-3 bg-rose-50 border border-rose-200 rounded-lg text-sm text-rose-700">{job.error}</div>
          )}
        </div>

        {/* Config */}
        <div className="rounded-xl border border-gray-200 p-6 mb-6">
          <p className="text-xs text-gray-400 uppercase tracking-wide mb-3">Configuration</p>
          <div className="grid grid-cols-2 gap-y-2 text-sm">
            {isGT && (
              <div><span className="text-gray-500">Environment:</span> <span className="text-gray-900">{envIcon} {envLabel}</span></div>
            )}
            <div><span className="text-gray-500">Dataset:</span> <span className="text-gray-900">{job.config.dataset}</span></div>
            {!isGT && <div><span className="text-gray-500">Model:</span> <span className="text-gray-900">{job.config.model}</span></div>}
            <div><span className="text-gray-500">Scenarios:</span> <span className="text-gray-900">{numScenes}</span></div>
            {!isGT && <div><span className="text-gray-500">Views:</span> <span className="text-gray-900">{maxViews}</span></div>}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          {job.status === "completed" && isGT && job.dataset_id && (
            <Link href="/datasets" className="flex-1 py-2.5 bg-sky-500 rounded-lg hover:bg-sky-600 text-white text-sm font-medium text-center transition">
              View Dataset
            </Link>
          )}
          {job.status === "completed" && job.projectId && (
            <Link href={`/annotate?projectId=${job.projectId}&review=true`} className="flex-1 py-2.5 bg-sky-500 rounded-lg hover:bg-sky-600 text-white text-sm font-medium text-center transition">
              View Results
            </Link>
          )}
          {job.status === "queued" && (
            <button
              onClick={async () => { const res = await fetch(`/api/jobs/${id}/run`, { method: "POST" }); if (!res.ok) { const d = await res.json(); alert(d.detail || "Failed"); } }}
              className="flex-1 py-2.5 bg-sky-500 rounded-lg hover:bg-sky-600 text-white text-sm font-medium text-center transition"
            >
              Start Processing
            </button>
          )}
          {(job.status === "downloading" || job.status === "generating") && (
            <div className="flex-1 py-2.5 bg-gray-50 rounded-lg text-center text-gray-400 text-sm animate-pulse">Processing...</div>
          )}
          <Link href="/" className="px-5 py-2.5 border border-gray-200 rounded-lg text-gray-600 text-sm hover:bg-gray-50 transition">Home</Link>
        </div>
      </div>
    </div>
  );
}
