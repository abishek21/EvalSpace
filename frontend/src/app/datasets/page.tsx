"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

type Scenario = {
  id: string;
  scene_id: string;
  prompt: string;
  difficulty: string;
  ground_truth: {
    answer: string;
    stable: boolean;
    objects: { color: string; label: string; type: string }[];
    before_images: string[];
    after_images: string[];
  };
};

type Dataset = {
  id: string;
  name: string;
  scenario_count: number;
  created_at: string;
  task_type?: string;
  scenarios?: Scenario[];
};

const ENV_BADGE: Record<string, { icon: string; label: string }> = {
  stacking_stability: { icon: "🏗️", label: "Stacking" },
  collision_prediction: { icon: "🎱", label: "Collision" },
  spatial_fitting: { icon: "🔲", label: "Spatial Fitting" },
};

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<Dataset | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [viewIdx, setViewIdx] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/datasets").then(r => r.json()).then(setDatasets).catch(() => {});
  }, []);

  async function selectDataset(ds: Dataset) {
    setSelected(ds);
    setViewIdx(0);
    setLoading(true);
    const res = await fetch(`/api/datasets/${ds.id}`);
    const data = await res.json();
    setScenarios(data.scenarios || []);
    setLoading(false);
  }

  const sc = scenarios[viewIdx];

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
            <Link href="/datasets" className="text-sky-600 font-medium">Datasets</Link>
            <Link href="/generate" className="hover:text-gray-900">Generate</Link>
            <Link href="/evaluate" className="hover:text-gray-900">Evaluate</Link>
            <Link href="/rlhf" className="hover:text-gray-900">RLHF</Link>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 mb-1">Ground Truth Datasets</h1>
            <p className="text-sm text-gray-500">Inspect scenarios, images, and questions from your datasets</p>
          </div>
          <Link href="/generate" className="px-4 py-2 text-sm rounded-lg bg-sky-500 text-white hover:bg-sky-600 transition">
            + New Dataset
          </Link>
        </div>

        {datasets.length === 0 ? (
          <div className="text-center py-20 text-gray-400">
            <p className="text-lg mb-2">No datasets yet</p>
            <p className="text-sm">Go to <Link href="/generate" className="text-sky-500 underline">Generate</Link> to create your first ground truth dataset</p>
          </div>
        ) : (
          <div className="grid grid-cols-[280px_1fr] gap-6 min-h-[500px]">
            {/* Left: dataset list */}
            <div className="space-y-2">
              {datasets.map(ds => (
                <div key={ds.id} className={`relative group w-full text-left px-4 py-3 rounded-xl border transition ${
                    selected?.id === ds.id
                      ? "border-sky-300 bg-sky-50"
                      : "border-gray-200 hover:border-gray-300"
                  }`}>
                  <button onClick={() => selectDataset(ds)} className="w-full text-left">
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-gray-900 text-sm truncate pr-6">{ds.name}</p>
                      {ds.task_type && ENV_BADGE[ds.task_type] && (
                        <span className="text-xs px-1.5 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-medium shrink-0">
                          {ENV_BADGE[ds.task_type].icon} {ENV_BADGE[ds.task_type].label}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">{ds.scenario_count} scenarios · {new Date(ds.created_at).toLocaleDateString()}</p>
                  </button>
                  <button onClick={async (e) => { e.stopPropagation(); if (!confirm(`Delete "${ds.name}"?`)) return; await fetch(`/api/datasets/${ds.id}`, { method: "DELETE" }); setDatasets(prev => prev.filter(d => d.id !== ds.id)); if (selected?.id === ds.id) { setSelected(null); setScenarios([]); } }}
                    className="absolute top-2.5 right-2.5 p-1 rounded-md text-gray-300 hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-all"
                    title="Delete dataset">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                  </button>
                </div>
              ))}
            </div>

            {/* Right: scenario viewer */}
            <div className="rounded-xl border border-gray-200 overflow-hidden">
              {!selected ? (
                <div className="flex items-center justify-center h-full text-gray-400 text-sm">
                  Select a dataset to inspect
                </div>
              ) : loading ? (
                <div className="flex items-center justify-center h-full text-gray-400 text-sm">Loading...</div>
              ) : scenarios.length === 0 ? (
                <div className="flex items-center justify-center h-full text-gray-400 text-sm">No scenarios found</div>
              ) : (
                <div className="flex flex-col h-full">
                  {/* Scenario nav */}
                  <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100 bg-gray-50/50">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-gray-700">{selected.name}</span>
                      {selected.task_type && ENV_BADGE[selected.task_type] && (
                        <span className="text-xs px-1.5 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-medium">
                          {ENV_BADGE[selected.task_type].icon} {ENV_BADGE[selected.task_type].label}
                        </span>
                      )}
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        sc?.ground_truth?.answer === "stable" || sc?.ground_truth?.answer === "fits" || sc?.ground_truth?.answer === "hit" || sc?.ground_truth?.stable
                          ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                          : "bg-rose-50 text-rose-700 border border-rose-200"
                      }`}>
                        {selected.task_type === "spatial_fitting"
                          ? (sc?.ground_truth?.answer === "fits" ? "FITS" : "DOES NOT FIT")
                          : (sc?.ground_truth?.answer || (sc?.ground_truth?.stable ? "stable" : "unstable") || "").toUpperCase()}
                      </span>
                      {sc?.difficulty && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 capitalize">{sc.difficulty}</span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <button onClick={() => setViewIdx(Math.max(0, viewIdx - 1))} disabled={viewIdx <= 0}
                        className="p-1.5 rounded-lg hover:bg-gray-200 disabled:opacity-30 transition">
                        <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" /></svg>
                      </button>
                      <span className="text-sm text-gray-500 tabular-nums min-w-[50px] text-center">{viewIdx + 1} / {scenarios.length}</span>
                      <button onClick={() => setViewIdx(Math.min(scenarios.length - 1, viewIdx + 1))} disabled={viewIdx >= scenarios.length - 1}
                        className="p-1.5 rounded-lg hover:bg-gray-200 disabled:opacity-30 transition">
                        <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>
                      </button>
                    </div>
                  </div>

                  {/* Scenario content */}
                  <div className="flex-1 p-5 overflow-y-auto space-y-5">
                    {/* Question */}
                    <div>
                      <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Question</p>
                      <p className="text-sm text-gray-800">{sc?.prompt || "—"}</p>
                    </div>

                    {/* Objects */}
                    {selected.task_type !== "spatial_fitting" && sc?.ground_truth?.objects && (
                      <div>
                        <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Stack (bottom → top)</p>
                        <div className="flex flex-wrap gap-1.5">
                          {sc.ground_truth.objects.map((o, i) => (
                            <span key={i} className="text-xs bg-gray-100 border border-gray-200 px-2 py-1 rounded-lg">
                              {i + 1}. {o.color} {o.label}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Images */}
                    <div className={`grid ${selected.task_type === "spatial_fitting" ? "grid-cols-1" : "grid-cols-2"} gap-4`}>
                      <div>
                        <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">{selected.task_type === "spatial_fitting" ? "Scene" : "Before"}</p>
                        <div className="grid grid-cols-2 gap-1.5">
                          {(sc?.ground_truth?.before_images || []).slice(0, 4).map((img, i) => (
                            <img key={i} src={img} alt="" className="w-full rounded-lg border border-gray-100" />
                          ))}
                          {(!sc?.ground_truth?.before_images || sc.ground_truth.before_images.length === 0) && (
                            <div className="col-span-2 h-28 bg-gray-50 rounded-lg flex items-center justify-center text-xs text-gray-400">No images</div>
                          )}
                        </div>
                      </div>
                      {selected.task_type !== "spatial_fitting" && (
                      <div>
                        <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">After (simulation)</p>
                        <div className="grid grid-cols-2 gap-1.5">
                          {(sc?.ground_truth?.after_images || []).slice(0, 4).map((img, i) => (
                            <img key={i} src={img} alt="" className="w-full rounded-lg border border-gray-100" />
                          ))}
                          {(!sc?.ground_truth?.after_images || sc.ground_truth.after_images.length === 0) && (
                            <div className="col-span-2 h-28 bg-gray-50 rounded-lg flex items-center justify-center text-xs text-gray-400">No images</div>
                          )}
                        </div>
                      </div>
                      )}
                    </div>

                    {/* Scene ID */}
                    <div className="text-xs text-gray-400 pt-2 border-t border-gray-100">
                      Scene: {sc?.scene_id || sc?.id?.slice(0, 8) || "—"}
                    </div>
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
