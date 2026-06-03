"use client";

import { useState, useEffect, useCallback, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

type Pair = {
  id: string;
  prompt: string;
  chosen: string;
  rejected: string;
  category?: string;
  difficulty?: string;
  source?: string | { dataset: string; images?: string[] };
  generation?: string | { model: string; task_type?: string };
  verification?: string | { mujoco_ground_truth: string; vlm_prediction: string; vlm_reasoning: string };
  ground_truth?: string | { stable: boolean; answer?: string; objects: { type: string; color: string; label: string; role?: string }[]; after_images?: string[]; before_images?: string[]; frames?: string[] };
  scene_id?: string;
  pair_type?: string;
  prediction?: string;
  reasoning?: string;
  correct?: boolean;
  model_response?: string;
};

/* ── Simulation Video Player ── */
function SimulationPlayer({ frames }: { frames: string[] }) {
  const [playing, setPlaying] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);
  const [speed, setSpeed] = useState(1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (playing) {
      // Base: 50ms per frame = 20fps. Speed 0.5x = 100ms, 2x = 25ms
      const ms = Math.round(50 / speed);
      intervalRef.current = setInterval(() => {
        setFrameIdx(prev => {
          if (prev >= frames.length - 1) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, ms);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, frames.length, speed]);

  function togglePlay() {
    if (frameIdx >= frames.length - 1) setFrameIdx(0);
    setPlaying(!playing);
  }

  function restart() {
    setFrameIdx(0);
    setPlaying(true);
  }

  const progress = frames.length > 1 ? (frameIdx / (frames.length - 1)) * 100 : 0;

  return (
    <div className="rounded-xl border border-gray-200 overflow-hidden bg-gray-900">
      <div className="relative">
        <img src={frames[frameIdx]} alt="" className="w-full aspect-[4/3] object-contain bg-gray-900" />
        {/* Frame indicator overlay */}
        {!playing && frameIdx === 0 && (
          <button onClick={togglePlay} className="absolute inset-0 flex items-center justify-center bg-black/20 hover:bg-black/30 transition group">
            <div className="w-14 h-14 rounded-full bg-white/90 group-hover:bg-white flex items-center justify-center shadow-lg">
              <svg className="w-6 h-6 text-gray-800 ml-1" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.84A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.27l9.344-5.891a1.5 1.5 0 000-2.538L6.3 2.841z" /></svg>
            </div>
          </button>
        )}
      </div>
      {/* Progress bar */}
      <div className="h-1 bg-gray-800 relative">
        <div className="h-full bg-sky-400 transition-all duration-75" style={{ width: `${progress}%` }} />
      </div>
      {/* Controls */}
      <div className="flex items-center gap-3 px-3 py-2 bg-gray-800">
        <button onClick={togglePlay} className="p-1 rounded hover:bg-gray-700 transition text-white" title={playing ? "Pause" : "Play"}>
          {playing ? (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M5.75 3a.75.75 0 01.75.75v12.5a.75.75 0 01-1.5 0V3.75A.75.75 0 015.75 3zm8.5 0a.75.75 0 01.75.75v12.5a.75.75 0 01-1.5 0V3.75a.75.75 0 01.75-.75z" /></svg>
          ) : (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.84A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.27l9.344-5.891a1.5 1.5 0 000-2.538L6.3 2.841z" /></svg>
          )}
        </button>
        <button onClick={restart} className="p-1 rounded hover:bg-gray-700 transition text-gray-400 hover:text-white" title="Replay">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" /></svg>
        </button>
        <input
          type="range" min={0} max={frames.length - 1} value={frameIdx}
          onChange={e => { setFrameIdx(Number(e.target.value)); setPlaying(false); }}
          className="flex-1 h-1 accent-sky-400 cursor-pointer"
        />
        <span className="text-xs text-gray-400 tabular-nums min-w-[40px] text-right">{frameIdx + 1}/{frames.length}</span>
        <select
          value={speed} onChange={e => setSpeed(Number(e.target.value))}
          className="text-xs bg-gray-700 text-gray-300 rounded px-1.5 py-0.5 border-none outline-none cursor-pointer"
        >
          <option value={0.25}>0.25×</option>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
        </select>
      </div>
    </div>
  );
}

function parse(val: unknown): Record<string, unknown> | null {
  if (!val) return null;
  if (typeof val === "string") { try { return JSON.parse(val); } catch { return null; } }
  return val as Record<string, unknown>;
}

/* ── Saved toast ── */
function SavedToast({ show, message }: { show: boolean; message: string }) {
  return (
    <div className={`fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-2.5 rounded-lg shadow-lg border transition-all duration-300 ${
      show ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-3 pointer-events-none"
    } bg-white border-emerald-200`}>
      <svg className="w-4 h-4 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
      </svg>
      <span className="text-sm font-medium text-gray-700">{message}</span>
    </div>
  );
}

function AnnotateInner() {
  const searchParams = useSearchParams();
  const projectId = searchParams.get("projectId");
  const review = searchParams.get("review") === "true";
  const pairType = searchParams.get("type") || "project";
  const [pair, setPair] = useState<Pair | null>(null);
  const [done, setDone] = useState(false);
  const [humanReasoning, setHumanReasoning] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [idx, setIdx] = useState(0);
  const [total, setTotal] = useState(0);
  const [toast, setToast] = useState({ show: false, message: "" });

  function showToast(message: string) {
    setToast({ show: true, message });
    setTimeout(() => setToast({ show: false, message: "" }), 1800);
  }

  const load = useCallback(async (i?: number) => {
    if (!projectId) return;
    const index = i ?? idx;
    const params = review || pairType === "eval"
      ? `projectId=${projectId}&review=true&index=${index}&type=${pairType}`
      : `projectId=${projectId}&type=${pairType}`;
    const res = await fetch(`/api/pairs?${params}`);
    const data = await res.json();
    if (data.done) { setDone(true); setPair(null); }
    else { setPair(data.pair); setHumanReasoning(""); }
    if (data.total !== undefined) setTotal(data.total);
  }, [projectId, review, idx, pairType]);

  useEffect(() => { setIdx(0); setDone(false); load(0); }, [projectId, review]); // eslint-disable-line

  async function submit(preference: string, label: string) {
    if (!pair) return;
    setSubmitting(true);
    await fetch("/api/pairs", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: pair.id, preference, rationale: humanReasoning }),
    });
    setSubmitting(false);
    showToast(label);
    // Auto-advance after short delay so user sees the toast
    setTimeout(() => {
      const next = idx + 1;
      setIdx(next);
      load(next);
    }, 600);
  }

  function nav(dir: number) {
    const next = Math.max(0, Math.min(total - 1, idx + dir));
    setIdx(next);
    load(next);
  }

  function skip() {
    const next = idx + 1;
    setIdx(next);
    load(next);
  }

  if (!projectId) return <div className="min-h-screen bg-white flex items-center justify-center text-gray-400">No projectId</div>;

  if (done) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center max-w-sm">
          <div className="w-12 h-12 bg-sky-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-6 h-6 text-sky-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
            </svg>
          </div>
          <p className="text-lg font-semibold text-gray-900 mb-1">All done</p>
          <p className="text-sm text-gray-500 mb-6">{total} scenarios reviewed</p>
          <div className="space-y-2">
            <Link href={`/annotate?projectId=${projectId}&review=true&type=${pairType}`} className="block px-4 py-2 rounded-lg bg-sky-500 text-white text-sm hover:bg-sky-600 transition">
              Review all
            </Link>
            <Link href="/" className="block px-4 py-2 text-sm text-gray-500 hover:text-gray-900">Home</Link>
          </div>
        </div>
      </div>
    );
  }

  if (!pair) return <div className="min-h-screen bg-white flex items-center justify-center text-gray-400 text-sm">Loading...</div>;

  const v = parse(pair.verification) as { mujoco_ground_truth: string; vlm_prediction: string; vlm_reasoning: string } | null;
  const gt = parse(pair.ground_truth) as { stable: boolean; answer?: string; objects: { color: string; label: string; role?: string }[]; after_images?: string[]; before_images?: string[]; frames?: string[] } | null;
  const src = parse(pair.source) as { dataset: string; images?: string[] } | null;

  // Handle eval_result pairs from Stage 2
  const isEvalResult = pair.pair_type === "eval_result";
  const isCollision = pair.category === "collision_prediction";

  // Normalize: build a unified view for both old and eval_result formats
  const effectiveV = v || (isEvalResult ? {
    mujoco_ground_truth: gt?.answer || (gt?.stable ? "stable" : "unstable") || "",
    vlm_prediction: pair.prediction || "",
    vlm_reasoning: pair.reasoning || pair.model_response || "",
  } : null);

  const beforeImages = src?.images || gt?.before_images || [];
  const afterImages = gt?.after_images || [];
  const simFrames = gt?.frames || [];

  const isStacking = !isCollision && (pair.category === "stacking_stability" || !!effectiveV);
  const isCorrect = isEvalResult ? pair.correct : (effectiveV ? effectiveV.mujoco_ground_truth === effectiveV.vlm_prediction : null);

  if (isCollision && effectiveV) {
    const gtAnswer = effectiveV.mujoco_ground_truth.toUpperCase();
    const vlmAnswer = effectiveV.vlm_prediction.toUpperCase();
    return (
      <div className="h-screen bg-white flex flex-col overflow-hidden">
        <SavedToast show={toast.show} message={toast.message} />

        {/* Top bar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-gray-100 shrink-0">
          <div className="flex items-center gap-4">
            <Link href={pairType === "eval" ? "/evaluate" : "/"} className="text-gray-400 hover:text-gray-900">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
              </svg>
            </Link>
            <span className="text-sm text-gray-500">🎱 {pair.scene_id || `Scenario ${idx + 1}`}</span>
            {isCorrect !== null && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                isCorrect ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-rose-50 text-rose-700 border border-rose-200"
              }`}>
                {isCorrect ? "✓ Correct" : "✗ Incorrect"}
              </span>
            )}
            {pair.difficulty && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 capitalize">{pair.difficulty}</span>
            )}
          </div>
          {total > 0 && (
            <div className="flex items-center gap-2">
              <button onClick={() => nav(-1)} disabled={idx <= 0} className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition">
                <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" /></svg>
              </button>
              <span className="text-sm text-gray-500 tabular-nums min-w-[60px] text-center">{idx + 1} / {total}</span>
              <button onClick={() => nav(1)} disabled={idx >= total - 1} className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition">
                <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>
              </button>
            </div>
          )}
        </div>

        {/* Main content */}
        <div className="flex-1 grid grid-cols-[1fr_320px] gap-0 min-h-0">
          {/* Left: visuals + GT/VLM */}
          <div className="p-5 flex flex-col gap-5 overflow-y-auto border-r border-gray-100">
            {/* Question */}
            <p className="text-sm text-gray-700 font-medium">{pair.prompt}</p>

            {/* Row 1: Before images (left) + Video playback (right) */}
            <div className="grid grid-cols-2 gap-4">
              {/* Before images — multi-view */}
              <div>
                <p className="text-xs text-gray-400 mb-2 uppercase tracking-wide font-medium">📷 Initial Setup</p>
                <div className="grid grid-cols-2 gap-1.5">
                  {beforeImages.slice(0, 4).map((img: string, i: number) => (
                    <img key={i} src={img} alt="" className="w-full rounded-lg border border-gray-100 aspect-[4/3] object-cover" />
                  ))}
                  {beforeImages.length === 0 && (
                    <div className="col-span-2 h-40 bg-gray-50 rounded-lg flex items-center justify-center text-xs text-gray-400">No images</div>
                  )}
                </div>
              </div>

              {/* Video playback */}
              <div>
                <p className="text-xs text-gray-400 mb-2 uppercase tracking-wide font-medium">🎬 Simulation Playback</p>
                {simFrames.length > 0 ? (
                  <SimulationPlayer frames={simFrames} />
                ) : (
                  <div className="h-40 bg-gray-50 rounded-lg flex items-center justify-center text-xs text-gray-400">No simulation frames</div>
                )}
              </div>
            </div>

            {/* Row 2: GT vs VLM */}
            <div className="grid grid-cols-2 gap-4">
              <div className={`rounded-xl p-4 border ${isCorrect ? "border-emerald-200 bg-emerald-50/30" : "border-gray-200"}`}>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-2 font-medium">MuJoCo Ground Truth</p>
                <p className={`text-lg font-semibold ${gtAnswer === "HIT" ? "text-rose-600" : "text-emerald-600"}`}>
                  {gtAnswer === "HIT" ? "💥 HIT" : "✦ MISS"}
                </p>
                {gt?.objects && (
                  <div className="mt-3 flex flex-wrap gap-1">
                    {gt.objects.map((o: { color: string; label: string; role?: string }, i: number) => (
                      <span key={i} className={`text-xs px-1.5 py-0.5 rounded border ${
                        o.role === "pushed" ? "bg-orange-50 border-orange-200 text-orange-700"
                          : o.role === "target" ? "bg-blue-50 border-blue-200 text-blue-700"
                          : "bg-white border-gray-200 text-gray-600"
                      }`}>
                        {o.role === "pushed" ? "→ " : o.role === "target" ? "◎ " : "▪ "}{o.color} {o.label}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className={`rounded-xl p-4 border ${isCorrect ? "border-emerald-200 bg-emerald-50/30" : "border-rose-200 bg-rose-50/30"}`}>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-2 font-medium">VLM Prediction</p>
                <p className={`text-lg font-semibold ${vlmAnswer === "HIT" ? "text-rose-600" : "text-emerald-600"}`}>
                  {vlmAnswer === "HIT" ? "💥 HIT" : vlmAnswer === "MISS" ? "✦ MISS" : vlmAnswer}
                </p>
                <p className="text-xs text-gray-500 mt-3 leading-relaxed line-clamp-4">{effectiveV.vlm_reasoning}</p>
              </div>
            </div>
          </div>

          {/* Right sidebar — annotation controls */}
          <div className="p-4 flex flex-col bg-gray-50/50 gap-3">
            {review ? (
              <div className="flex-1 flex flex-col gap-4">
                <div className={`rounded-lg px-3 py-2 text-center text-sm font-medium ${
                  isCorrect ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-rose-50 text-rose-700 border border-rose-200"
                }`}>
                  {isCorrect ? "✓ VLM was correct" : "✗ VLM was incorrect"}
                </div>
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">VLM Reasoning</p>
                  <p className="text-sm text-gray-700 leading-relaxed">{effectiveV.vlm_reasoning}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Objects</p>
                  <div className="space-y-1">
                    {gt?.objects?.map((o: { color: string; label: string; role?: string }, i: number) => (
                      <div key={i} className="flex items-center gap-2 text-sm text-gray-600">
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          o.role === "pushed" ? "bg-orange-100 text-orange-700"
                            : o.role === "target" ? "bg-blue-100 text-blue-700"
                            : "bg-gray-100 text-gray-500"
                        }`}>{o.role || "obj"}</span>
                        <span>{o.color} {o.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Status + primary action at top */}
                <div className={`rounded-lg px-3 py-2.5 text-center text-sm font-medium ${
                  isCorrect ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-rose-50 text-rose-700 border border-rose-200"
                }`}>
                  {isCorrect ? "✓ VLM prediction matches ground truth" : "✗ VLM prediction doesn't match"}
                </div>

                <button
                  onClick={() => submit(
                    isCorrect ? "chosen" : "rejected",
                    isCorrect ? "Confirmed correct ✓" : "Confirmed incorrect ✗"
                  )}
                  disabled={submitting}
                  className={`w-full py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50 active:scale-[0.98] ${
                    isCorrect
                      ? "bg-emerald-500 text-white hover:bg-emerald-600"
                      : "bg-rose-500 text-white hover:bg-rose-600"
                  }`}
                >
                  {submitting ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                      Saving...
                    </span>
                  ) : isCorrect ? "✓ Confirm correct" : "✗ Confirm incorrect"}
                </button>

                {/* Reasoning */}
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Add reasoning (optional)</p>
                  <textarea
                    className="w-full h-24 rounded-lg border border-gray-200 p-2.5 text-sm resize-none focus:border-sky-400 focus:ring-1 focus:ring-sky-400 outline-none bg-white"
                    placeholder={isCorrect
                      ? "Notes (e.g. why VLM's reasoning is good)"
                      : `Why does the object ${gtAnswer === "HIT" ? "hit" : "miss"}? This trains the model.`
                    }
                    value={humanReasoning}
                    onChange={(e) => setHumanReasoning(e.target.value)}
                  />
                </div>

                {/* Override + Skip */}
                <div className="mt-auto space-y-2 pt-2 border-t border-gray-200">
                  <button
                    onClick={() => submit(
                      isCorrect ? "rejected" : "chosen",
                      isCorrect ? "Overridden → marked incorrect" : "Overridden → marked correct"
                    )}
                    disabled={submitting}
                    className="w-full py-2 rounded-lg text-xs border border-amber-300 text-amber-700 bg-amber-50 hover:bg-amber-100 transition disabled:opacity-50 active:scale-[0.98]"
                  >
                    {isCorrect ? "⚑ Override: VLM is actually wrong" : "⚑ Override: VLM is actually right"}
                  </button>

                  <button
                    onClick={skip}
                    className="w-full py-1.5 rounded-lg text-xs text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition"
                  >
                    Skip →
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (isStacking && effectiveV) {
    return (
      <div className="h-screen bg-white flex flex-col overflow-hidden">
        <SavedToast show={toast.show} message={toast.message} />

        {/* Top bar — always show nav */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-gray-100 shrink-0">
          <div className="flex items-center gap-4">
            <Link href={pairType === "eval" ? "/evaluate" : "/"} className="text-gray-400 hover:text-gray-900">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
              </svg>
            </Link>
            <span className="text-sm text-gray-500">{pair.scene_id || `Scenario ${idx + 1}`}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              isCorrect ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-rose-50 text-rose-700 border border-rose-200"
            }`}>
              {isCorrect ? "✓ Correct" : "✗ Incorrect"}
            </span>
            {pair.difficulty && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 capitalize">{pair.difficulty}</span>
            )}
          </div>
          {/* Always show nav when total > 0 */}
          {total > 0 && (
            <div className="flex items-center gap-2">
              <button onClick={() => nav(-1)} disabled={idx <= 0} className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition">
                <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" /></svg>
              </button>
              <span className="text-sm text-gray-500 tabular-nums min-w-[60px] text-center">{idx + 1} / {total}</span>
              <button onClick={() => nav(1)} disabled={idx >= total - 1} className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition">
                <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>
              </button>
            </div>
          )}
        </div>

        {/* Main content */}
        <div className="flex-1 grid grid-cols-[1fr_320px] gap-0 min-h-0">
          {/* Left: images + GT/VLM */}
          <div className="p-5 flex flex-col gap-4 overflow-y-auto border-r border-gray-100">
            <p className="text-sm text-gray-700">{pair.prompt}</p>

            {/* Before / After images */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-400 mb-2 uppercase tracking-wide">Before</p>
                <div className="grid grid-cols-2 gap-1.5">
                  {beforeImages.slice(0, 4).map((img: string, i: number) => (
                    <img key={i} src={img} alt="" className="w-full rounded-lg border border-gray-100" />
                  ))}
                  {beforeImages.length === 0 && (
                    <div className="col-span-2 h-32 bg-gray-50 rounded-lg flex items-center justify-center text-xs text-gray-400">No images</div>
                  )}
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-2 uppercase tracking-wide">After (3s sim)</p>
                <div className="grid grid-cols-2 gap-1.5">
                  {afterImages.slice(0, 4).map((img: string, i: number) => (
                    <img key={i} src={img} alt="" className="w-full rounded-lg border border-gray-100" />
                  ))}
                  {afterImages.length === 0 && (
                    <div className="col-span-2 h-32 bg-gray-50 rounded-lg flex items-center justify-center text-xs text-gray-400">No after images</div>
                  )}
                </div>
              </div>
            </div>

            {/* GT vs VLM side by side */}
            <div className="grid grid-cols-2 gap-4">
              <div className={`rounded-xl p-4 border ${isCorrect ? "border-emerald-200 bg-emerald-50/30" : "border-gray-200"}`}>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">MuJoCo Ground Truth</p>
                <p className={`text-sm font-medium ${effectiveV.mujoco_ground_truth === "stable" ? "text-emerald-700" : "text-rose-700"}`}>
                  {effectiveV.mujoco_ground_truth.toUpperCase()}
                </p>
                {gt?.objects && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {gt.objects.map((o: { color: string; label: string }, i: number) => (
                      <span key={i} className="text-xs bg-white border border-gray-200 px-1.5 py-0.5 rounded">{o.color} {o.label}</span>
                    ))}
                  </div>
                )}
              </div>
              <div className={`rounded-xl p-4 border ${isCorrect ? "border-emerald-200 bg-emerald-50/30" : "border-rose-200 bg-rose-50/30"}`}>
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">VLM Prediction</p>
                <p className={`text-sm font-medium ${effectiveV.vlm_prediction === "stable" ? "text-emerald-700" : "text-rose-700"}`}>
                  {effectiveV.vlm_prediction.toUpperCase()}
                </p>
                <p className="text-xs text-gray-500 mt-2 line-clamp-4">{effectiveV.vlm_reasoning}</p>
              </div>
            </div>
          </div>

          {/* Right sidebar */}
          <div className="p-5 flex flex-col bg-gray-50/50">
            {review ? (
              /* Review mode */
              <div className="flex-1 flex flex-col gap-4">
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">VLM Full Reasoning</p>
                  <p className="text-sm text-gray-700 leading-relaxed">{effectiveV.vlm_reasoning}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Stack Details</p>
                  <div className="space-y-1.5">
                    {gt?.objects?.map((o: { color: string; label: string }, i: number) => (
                      <div key={i} className="flex items-center gap-2 text-sm text-gray-600">
                        <span className="w-5 text-center text-xs text-gray-400">{i + 1}</span>
                        <span>{o.color} {o.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="mt-auto pt-4 border-t border-gray-200">
                  <p className="text-xs text-gray-400 mb-1">Difficulty</p>
                  <p className="text-sm text-gray-700 capitalize">{pair.difficulty || "—"}</p>
                </div>
              </div>
            ) : (
              /* Annotate mode */
              <>
                <div className="flex-1 flex flex-col">
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Your reasoning (optional)</p>
                  <textarea
                    className="flex-1 w-full rounded-xl border border-gray-200 p-3 text-sm resize-none focus:border-sky-400 focus:ring-1 focus:ring-sky-400 outline-none bg-white"
                    placeholder={isCorrect
                      ? "VLM was correct — add notes if needed"
                      : "Why is this stack " + effectiveV.mujoco_ground_truth + "? Your reasoning trains the model."
                    }
                    value={humanReasoning}
                    onChange={(e) => setHumanReasoning(e.target.value)}
                  />
                  {!isCorrect && (
                    <p className="text-xs text-rose-500 mt-2">
                      VLM was wrong — your reasoning becomes the training signal
                    </p>
                  )}
                </div>

                <div className="mt-4 space-y-2">
                  {/* Primary: Agree with GT */}
                  <button
                    onClick={() => submit(
                      isCorrect ? "chosen" : "rejected",
                      isCorrect ? "Confirmed correct ✓" : "Confirmed incorrect ✗"
                    )}
                    disabled={submitting}
                    className={`w-full py-2.5 rounded-lg text-sm font-medium transition disabled:opacity-50 active:scale-[0.98] ${
                      isCorrect
                        ? "bg-emerald-500 text-white hover:bg-emerald-600"
                        : "bg-rose-500 text-white hover:bg-rose-600"
                    }`}
                  >
                    {submitting ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                        Saving...
                      </span>
                    ) : isCorrect ? "✓ Confirm correct" : "✗ Confirm incorrect"}
                  </button>

                  {/* Override: Disagree with GT */}
                  <button
                    onClick={() => submit(
                      isCorrect ? "rejected" : "chosen",
                      isCorrect ? "Overridden → marked incorrect" : "Overridden → marked correct"
                    )}
                    disabled={submitting}
                    className="w-full py-2.5 rounded-lg text-sm border border-amber-300 text-amber-700 bg-amber-50 hover:bg-amber-100 transition disabled:opacity-50 active:scale-[0.98]"
                  >
                    {isCorrect ? "⚑ Override: I disagree, VLM is wrong" : "⚑ Override: I disagree, VLM is right"}
                  </button>

                  {/* Skip */}
                  <button
                    onClick={skip}
                    className="w-full py-2 rounded-lg text-xs text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition"
                  >
                    Skip → next scenario
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Fallback: generic RLHF pair view
  return (
    <div className="h-screen bg-white flex flex-col overflow-hidden">
      <SavedToast show={toast.show} message={toast.message} />
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-100 shrink-0">
        <Link href="/" className="text-gray-400 hover:text-gray-900">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
          </svg>
        </Link>
        {total > 0 && (
          <div className="flex items-center gap-2">
            <button onClick={() => nav(-1)} disabled={idx <= 0} className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition">
              <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" /></svg>
            </button>
            <span className="text-sm text-gray-500 tabular-nums">{idx + 1} / {total}</span>
            <button onClick={() => nav(1)} disabled={idx >= total - 1} className="p-1.5 rounded-lg hover:bg-gray-100 disabled:opacity-30 transition">
              <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>
            </button>
          </div>
        )}
      </div>

      <div className="flex-1 grid grid-cols-[1fr_300px] gap-0 min-h-0">
        <div className="p-6 overflow-y-auto space-y-4">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Prompt</p>
            <p className="text-sm text-gray-800">{pair.prompt}</p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-xl border border-gray-200 p-4">
              <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Answer A</p>
              <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans">{pair.chosen}</pre>
            </div>
            <div className="rounded-xl border border-gray-200 p-4">
              <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Answer B</p>
              <pre className="text-sm text-gray-700 whitespace-pre-wrap font-sans">{pair.rejected}</pre>
            </div>
          </div>
        </div>

        <div className="p-5 flex flex-col bg-gray-50/50 border-l border-gray-100">
          <div className="flex-1">
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Notes</p>
            <textarea
              className="w-full h-24 rounded-xl border border-gray-200 p-3 text-sm resize-none focus:border-sky-400 outline-none bg-white"
              placeholder="Optional rationale"
              value={humanReasoning}
              onChange={(e) => setHumanReasoning(e.target.value)}
            />
          </div>
          <div className="mt-4 space-y-2">
            <button onClick={() => submit("chosen", "Saved: A is better")} disabled={submitting} className="w-full py-2.5 rounded-lg text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-600 transition disabled:opacity-50 active:scale-[0.98]">
              A is better
            </button>
            <button onClick={() => submit("tie", "Saved: Tie")} disabled={submitting} className="w-full py-2.5 rounded-lg text-sm border border-gray-200 text-gray-600 hover:bg-gray-100 transition disabled:opacity-50 active:scale-[0.98]">
              Tie
            </button>
            <button onClick={() => submit("rejected", "Saved: B is better")} disabled={submitting} className="w-full py-2.5 rounded-lg text-sm font-medium bg-rose-500 text-white hover:bg-rose-600 transition disabled:opacity-50 active:scale-[0.98]">
              B is better
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function AnnotatePage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-white" />}>
      <AnnotateInner />
    </Suspense>
  );
}
