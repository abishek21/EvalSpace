"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type VerificationCheck = {
  type: string;
  claim: string;
  ground_truth?: number;
  correct: boolean;
};

type Verification = {
  physics_correct: boolean | null;
  checks: VerificationCheck[];
  ground_truth_summary: {
    object_count: number;
    categories: Record<string, number>;
    num_relations: number;
  };
};

type Pair = {
  id: string;
  prompt: string;
  chosen: string;
  rejected: string;
  category: string;
  difficulty: string;
  scene_id: string;
  status: string;
  preference?: string;
  source?: {
    dataset: string;
    scene_id: string;
    images: string[];
    ground_truth?: {
      object_count: number;
      objects: Record<string, { label: string; type: string; position: number[] }>;
      spatial_relations: string[];
      categories: Record<string, number>;
    };
  };
  generation?: {
    model: string;
    source_type?: string;
    num_views: number;
  };
  verification?: Verification;
};

function VerifyInner() {
  const searchParams = useSearchParams();
  const projectId = searchParams.get("projectId");
  const [pairs, setPairs] = useState<Pair[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showGroundTruth, setShowGroundTruth] = useState(false);

  useEffect(() => {
    if (!projectId) return;
    setLoading(true);
    fetch(`${API}/api/export?projectId=${projectId}&format=full&all=true`)
      .then((r) => r.json())
      .then((data) => {
        setPairs(data.data || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [projectId]);

  const pair = pairs[currentIdx];

  if (!projectId) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <p className="text-gray-400">No projectId provided. Go to <Link href="/" className="text-blue-400">Dashboard</Link></p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <p className="text-gray-400">Loading pairs...</p>
      </div>
    );
  }

  if (pairs.length === 0) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <p className="text-gray-400">No pairs found for this project.</p>
      </div>
    );
  }

  const gt = pair?.source?.ground_truth;
  const verification = pair?.verification;
  const isMujoco = pair?.source?.dataset?.startsWith("mujoco") || pair?.generation?.source_type === "mujoco";

  // Stats
  const verifiedCount = pairs.filter((p) => p.verification?.physics_correct === true).length;
  const failedCount = pairs.filter((p) => p.verification?.physics_correct === false).length;
  const unchecked = pairs.filter((p) => p.verification?.physics_correct == null).length;

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-[1800px] mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              🔬 Physics Verification
            </h1>
            <p className="text-gray-400 text-sm mt-1">
              MuJoCo ground truth vs VLM answers — pair {currentIdx + 1} of {pairs.length}
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* Stats badges */}
            <div className="flex gap-2 text-sm">
              <span className="px-3 py-1 bg-green-900/50 border border-green-700 rounded-full">
                ✅ {verifiedCount} correct
              </span>
              <span className="px-3 py-1 bg-red-900/50 border border-red-700 rounded-full">
                ❌ {failedCount} failed
              </span>
              <span className="px-3 py-1 bg-gray-800 border border-gray-700 rounded-full">
                ❓ {unchecked} unchecked
              </span>
            </div>
            <Link href="/" className="text-gray-400 hover:text-white text-sm">
              ← Dashboard
            </Link>
          </div>
        </div>

        {pair && (
          <div className="grid grid-cols-12 gap-4">
            {/* Left: Scene Views (4 cols) */}
            <div className="col-span-4 space-y-4">
              {/* Scene Images */}
              <div className="bg-gray-900 rounded-xl p-4">
                <h2 className="text-sm font-semibold text-gray-400 mb-3">📷 SCENE VIEWS</h2>
                {pair.source?.images && pair.source.images.length > 0 ? (
                  <div className="space-y-3">
                    {pair.source.images.map((img, i) => (
                      <div key={i} className="relative">
                        <img
                          src={img}
                          alt={`View ${i + 1}`}
                          className="w-full rounded-lg border border-gray-800"
                        />
                        <span className="absolute top-2 left-2 text-xs bg-black/80 px-2 py-1 rounded font-mono">
                          {["Front", "Top", "Side"][i] || `View ${i + 1}`}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="h-48 flex items-center justify-center border-2 border-dashed border-gray-800 rounded-lg">
                    <p className="text-gray-600">No images</p>
                  </div>
                )}
              </div>

              {/* Ground Truth Panel */}
              {gt && (
                <div className="bg-gray-900 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-semibold text-gray-400">🎯 GROUND TRUTH</h2>
                    <button
                      onClick={() => setShowGroundTruth(!showGroundTruth)}
                      className="text-xs text-blue-400 hover:text-blue-300"
                    >
                      {showGroundTruth ? "Hide" : "Show"} details
                    </button>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Objects:</span>
                      <span className="font-mono">{gt.object_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Types:</span>
                      <span className="font-mono text-xs">
                        {Object.entries(gt.categories).map(([k, v]) => `${v}× ${k}`).join(", ")}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Relations:</span>
                      <span className="font-mono">{gt.spatial_relations?.length || 0}</span>
                    </div>
                  </div>

                  {showGroundTruth && (
                    <div className="mt-3 space-y-3">
                      {/* Object positions */}
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Object Positions:</p>
                        {Object.entries(gt.objects || {}).map(([name, obj]) => (
                          <div key={name} className="flex justify-between text-xs font-mono py-0.5">
                            <span className="text-gray-300">{obj.label}</span>
                            <span className="text-gray-500">
                              ({obj.position.map((p: number) => p.toFixed(2)).join(", ")})
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* Spatial relations */}
                      <div>
                        <p className="text-xs text-gray-500 mb-1">Spatial Relations:</p>
                        <div className="max-h-32 overflow-y-auto space-y-0.5">
                          {gt.spatial_relations?.map((rel: string, i: number) => (
                            <p key={i} className="text-xs text-gray-400 font-mono">• {rel}</p>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Middle: Q&A (5 cols) */}
            <div className="col-span-5 space-y-4">
              {/* Question */}
              <div className="bg-gray-900 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs px-2 py-0.5 bg-blue-900 rounded">{pair.category}</span>
                  <span className="text-xs px-2 py-0.5 bg-purple-900 rounded">{pair.difficulty}</span>
                  {isMujoco && (
                    <span className="text-xs px-2 py-0.5 bg-orange-900 rounded">🤖 MuJoCo</span>
                  )}
                </div>
                <p className="text-lg">{pair.prompt}</p>
              </div>

              {/* Chosen Answer */}
              <div className={`bg-gray-900 rounded-xl p-4 border-2 ${
                verification?.physics_correct === true ? "border-green-600" :
                verification?.physics_correct === false ? "border-red-600" :
                "border-transparent"
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-semibold text-green-400">✅ CHOSEN (CoT + Images)</p>
                  {verification?.physics_correct === true && (
                    <span className="text-xs px-2 py-0.5 bg-green-900 text-green-300 rounded-full">
                      Physics ✓
                    </span>
                  )}
                  {verification?.physics_correct === false && (
                    <span className="text-xs px-2 py-0.5 bg-red-900 text-red-300 rounded-full">
                      Physics ✗
                    </span>
                  )}
                </div>
                <div className="max-h-72 overflow-y-auto">
                  <pre className="whitespace-pre-wrap text-sm text-gray-200 font-sans leading-relaxed">
                    {pair.chosen}
                  </pre>
                </div>
              </div>

              {/* Rejected Answer */}
              <div className="bg-gray-900 rounded-xl p-4 border-2 border-transparent opacity-75">
                <p className="text-sm font-semibold text-red-400 mb-2">❌ REJECTED (Text-only, no images)</p>
                <div className="max-h-40 overflow-y-auto">
                  <pre className="whitespace-pre-wrap text-sm text-gray-400 font-sans">
                    {pair.rejected}
                  </pre>
                </div>
              </div>
            </div>

            {/* Right: Verification (3 cols) */}
            <div className="col-span-3 space-y-4">
              {/* Verification Status */}
              <div className={`rounded-xl p-4 ${
                verification?.physics_correct === true ? "bg-green-950 border border-green-800" :
                verification?.physics_correct === false ? "bg-red-950 border border-red-800" :
                "bg-gray-900 border border-gray-800"
              }`}>
                <h2 className="text-sm font-semibold mb-3">
                  {verification?.physics_correct === true ? "🟢 PHYSICS VERIFIED" :
                   verification?.physics_correct === false ? "🔴 PHYSICS MISMATCH" :
                   "⚪ NOT VERIFIED"}
                </h2>

                {verification?.checks && verification.checks.length > 0 ? (
                  <div className="space-y-2">
                    {verification.checks.map((check, i) => (
                      <div key={i} className={`p-2 rounded text-sm ${
                        check.correct ? "bg-green-900/30" : "bg-red-900/30"
                      }`}>
                        <div className="flex items-center gap-2">
                          <span>{check.correct ? "✅" : "❌"}</span>
                          <span className="text-xs px-1.5 py-0.5 bg-gray-800 rounded uppercase">
                            {check.type}
                          </span>
                        </div>
                        <p className="text-xs text-gray-300 mt-1">{check.claim}</p>
                        {check.ground_truth != null && (
                          <p className="text-xs text-gray-500 mt-0.5">
                            Ground truth: {check.ground_truth}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">
                    {isMujoco
                      ? "No specific claims could be auto-verified for this question type."
                      : "Physics verification only available for MuJoCo-generated scenes."}
                  </p>
                )}
              </div>

              {/* Ground Truth Summary */}
              {verification?.ground_truth_summary && (
                <div className="bg-gray-900 rounded-xl p-4">
                  <h2 className="text-sm font-semibold text-gray-400 mb-2">📊 SUMMARY</h2>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="bg-gray-800 rounded-lg p-2">
                      <p className="text-xl font-bold">{verification.ground_truth_summary.object_count}</p>
                      <p className="text-xs text-gray-500">Objects</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-2">
                      <p className="text-xl font-bold">
                        {Object.keys(verification.ground_truth_summary.categories).length}
                      </p>
                      <p className="text-xs text-gray-500">Types</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-2">
                      <p className="text-xl font-bold">{verification.ground_truth_summary.num_relations}</p>
                      <p className="text-xs text-gray-500">Relations</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Annotation status */}
              <div className="bg-gray-900 rounded-xl p-4">
                <h2 className="text-sm font-semibold text-gray-400 mb-2">📝 ANNOTATION</h2>
                <p className="text-sm">
                  {pair.preference
                    ? <span className={pair.preference === "chosen" ? "text-green-400" : "text-red-400"}>
                        Preference: {pair.preference}
                      </span>
                    : <span className="text-gray-500">Not yet annotated</span>
                  }
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-center gap-4 mt-6">
          <button
            onClick={() => setCurrentIdx(Math.max(0, currentIdx - 1))}
            disabled={currentIdx === 0}
            className="px-6 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 disabled:opacity-30 transition"
          >
            ← Previous
          </button>

          {/* Quick jump */}
          <div className="flex gap-1">
            {pairs.slice(Math.max(0, currentIdx - 3), currentIdx + 4).map((p, i) => {
              const idx = Math.max(0, currentIdx - 3) + i;
              return (
                <button
                  key={idx}
                  onClick={() => setCurrentIdx(idx)}
                  className={`w-8 h-8 rounded text-xs font-mono transition ${
                    idx === currentIdx
                      ? "bg-blue-600 text-white"
                      : p.verification?.physics_correct === true
                        ? "bg-green-900/50 text-green-300 hover:bg-green-800"
                        : p.verification?.physics_correct === false
                          ? "bg-red-900/50 text-red-300 hover:bg-red-800"
                          : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                  }`}
                >
                  {idx + 1}
                </button>
              );
            })}
          </div>

          <button
            onClick={() => setCurrentIdx(Math.min(pairs.length - 1, currentIdx + 1))}
            disabled={currentIdx === pairs.length - 1}
            className="px-6 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 disabled:opacity-30 transition"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}

export default function VerifyPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-gray-950" />}>
      <VerifyInner />
    </Suspense>
  );
}
