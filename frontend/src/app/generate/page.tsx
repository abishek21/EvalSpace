"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

const OBJECTS = [
  { id: "large_box", label: "Large Box", icon: "📦" },
  { id: "small_box", label: "Small Box", icon: "📦" },
  { id: "wide_box", label: "Wide Box", icon: "📦" },
  { id: "tiny_box", label: "Tiny Box", icon: "📦" },
  { id: "book", label: "Book", icon: "📕" },
  { id: "flat_plate", label: "Flat Plate", icon: "🍽️" },
  { id: "tall_cylinder", label: "Cylinder", icon: "🧪" },
  { id: "sphere", label: "Ball", icon: "⚽" },
  { id: "small_sphere", label: "Small Ball", icon: "🔵" },
  { id: "bowl", label: "Bowl", icon: "🥣" },
];

type Step = 1 | 2 | 3 | 4;

export default function GeneratePage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>(1);
  const [environment, setEnvironment] = useState<"stacking_stability" | "collision_prediction">("stacking_stability");
  const [selectedObjects, setSelectedObjects] = useState<string[]>(OBJECTS.map((o) => o.id));
  const [numViews, setNumViews] = useState(4);
  const [numStable, setNumStable] = useState(5);
  const [numUnstable, setNumUnstable] = useState(5);
  const [mode, setMode] = useState<"curated" | "random">("random");
  const [datasetName, setDatasetName] = useState("");
  const [answerModel, setAnswerModel] = useState("none");
  const [questionSource, setQuestionSource] = useState<"template" | "azure">("template");
  const [azureEndpoint, setAzureEndpoint] = useState("https://akaudiobot.services.ai.azure.com/openai/v1");
  const [azureKey, setAzureKey] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function toggleObject(id: string) {
    setSelectedObjects((prev) =>
      prev.includes(id) ? prev.filter((o) => o !== id) : [...prev, id]
    );
  }

  async function handleCreate() {
    setSubmitting(true);
    const body = {
      dataset: environment === "collision_prediction" ? "mujoco:collision" : "mujoco:stacking",
      job_type: "generate_gt",
      environment,
      split: "train",
      numScenes: mode === "curated" ? 10 : numStable + numUnstable,
      questionsPerScene: 1,
      maxViews: numViews,
      imageResolution: 480,
      categories: ["stacking_stability"],
      model: answerModel,
      name: datasetName || `stacking-${Date.now()}`,
      useCurated: mode === "curated",
      ...(mode === "random" ? { numStable, numUnstable } : {}),
      ...((questionSource === "azure" || answerModel === "gpt-4o") && azureKey ? {
        azureConfig: {
          endpoint: azureEndpoint,
          apiKey: azureKey,
        },
      } : {}),
      ...(questionSource === "azure" && azureKey ? {
        questionModel: {
          provider: "azure-openai",
          endpoint: azureEndpoint,
          apiKey: azureKey,
        },
      } : {}),
    };
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const job = await res.json();
    router.push(`/jobs/${job.id}`);
  }

  const total = mode === "curated" ? 10 : numStable + numUnstable;

  return (
    <div className="min-h-screen bg-white">
      {/* Nav */}
      <nav className="border-b border-gray-100 px-8 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-sky-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-sm font-bold">S</span>
            </div>
            <Link href="/" className="font-semibold text-gray-900">Spatial Reasoning Lab</Link>
          </div>
          <div className="flex items-center gap-6 text-sm text-gray-500">
            <Link href="/" className="hover:text-gray-900">Home</Link>
            <Link href="/datasets" className="hover:text-gray-900">Datasets</Link>
            <Link href="/generate" className="text-sky-600 font-medium">Generate</Link>
            <Link href="/evaluate" className="hover:text-gray-900">Evaluate</Link>
            <Link href="/rlhf" className="hover:text-gray-900">RLHF</Link>
          </div>
        </div>
      </nav>

      <div className="max-w-3xl mx-auto px-8 py-10">
        {/* Steps indicator */}
        <div className="flex items-center gap-0 mb-10">
          {[
            { n: 1, label: "Environment" },
            { n: 2, label: "Objects" },
            { n: 3, label: "Settings" },
            { n: 4, label: "Create" },
          ].map((s, i) => (
            <div key={s.n} className="flex items-center flex-1">
              <button
                onClick={() => s.n <= step && setStep(s.n as Step)}
                className={`flex items-center gap-2 ${s.n <= step ? "cursor-pointer" : "cursor-default"}`}
              >
                <div className={`w-7 h-7 rounded-full flex items-center justify-center text-sm font-medium transition ${
                  s.n === step ? "bg-sky-500 text-white" : s.n < step ? "bg-sky-100 text-sky-600" : "bg-gray-100 text-gray-400"
                }`}>
                  {s.n < step ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                    </svg>
                  ) : s.n}
                </div>
                <span className={`text-sm ${s.n === step ? "text-gray-900 font-medium" : "text-gray-400"}`}>
                  {s.label}
                </span>
              </button>
              {i < 3 && <div className={`flex-1 h-px mx-3 ${s.n < step ? "bg-sky-200" : "bg-gray-100"}`} />}
            </div>
          ))}
        </div>

        {/* Step 1: Environment */}
        {step === 1 && (
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-1">Choose environment</h2>
            <p className="text-sm text-gray-500 mb-6">Select the spatial reasoning task to generate ground truths for</p>

            <div className="space-y-3">
              <button
                onClick={() => { setEnvironment("stacking_stability"); setStep(2); }}
                className={`w-full text-left p-5 rounded-xl border-2 transition hover:shadow-sm ${
                  environment === "stacking_stability" ? "border-sky-200 bg-sky-50/50" : "border-gray-100 hover:border-gray-200"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">📦</span>
                      <p className="font-medium text-gray-900">Stacking Stability</p>
                    </div>
                    <p className="text-sm text-gray-500 mt-0.5 ml-7">
                      Will a stack of objects remain stable or topple? MuJoCo simulates for 3 seconds.
                    </p>
                  </div>
                  <svg className="w-5 h-5 text-sky-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                  </svg>
                </div>
              </button>

              <button
                onClick={() => { setEnvironment("collision_prediction"); setStep(2); }}
                className={`w-full text-left p-5 rounded-xl border-2 transition hover:shadow-sm ${
                  environment === "collision_prediction" ? "border-sky-200 bg-sky-50/50" : "border-gray-100 hover:border-gray-200"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🎱</span>
                      <p className="font-medium text-gray-900">Collision Prediction</p>
                    </div>
                    <p className="text-sm text-gray-500 mt-0.5 ml-7">
                      If object A is pushed, will it hit object B? Supports obstacles, chain collisions, and animation playback.
                    </p>
                  </div>
                  <svg className="w-5 h-5 text-sky-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                  </svg>
                </div>
              </button>

              <div className="w-full text-left p-5 rounded-xl border border-gray-100 opacity-40">
                <div className="flex items-center gap-2">
                  <span className="text-lg">🌉</span>
                  <p className="font-medium text-gray-900">Bridge Support</p>
                </div>
                <p className="text-sm text-gray-400 mt-0.5 ml-7">Coming soon — Can a structure support a load?</p>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Objects */}
        {step === 2 && (
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-1">Select objects</h2>
            <p className="text-sm text-gray-500 mb-6">Choose which objects can appear in stacking scenarios</p>

            <div className="grid grid-cols-5 gap-3 mb-6">
              {OBJECTS.map((obj) => {
                const selected = selectedObjects.includes(obj.id);
                return (
                  <button
                    key={obj.id}
                    onClick={() => toggleObject(obj.id)}
                    className={`p-3 rounded-xl border-2 text-center transition ${
                      selected
                        ? "border-sky-300 bg-sky-50"
                        : "border-gray-100 hover:border-gray-200"
                    }`}
                  >
                    <span className="text-xl block mb-1">{obj.icon}</span>
                    <span className={`text-xs ${selected ? "text-gray-900" : "text-gray-400"}`}>
                      {obj.label}
                    </span>
                  </button>
                );
              })}
            </div>

            <div className="flex items-center justify-between text-sm text-gray-500 mb-8">
              <span>{selectedObjects.length} of {OBJECTS.length} selected</span>
              <button
                onClick={() => setSelectedObjects(
                  selectedObjects.length === OBJECTS.length ? [] : OBJECTS.map((o) => o.id)
                )}
                className="text-sky-600 hover:text-sky-700"
              >
                {selectedObjects.length === OBJECTS.length ? "Deselect all" : "Select all"}
              </button>
            </div>

            <div className="flex gap-3">
              <button onClick={() => setStep(1)} className="px-5 py-2.5 rounded-lg border border-gray-200 text-gray-700 text-sm hover:bg-gray-50">
                Back
              </button>
              <button
                onClick={() => setStep(3)}
                disabled={selectedObjects.length < 2}
                className="px-5 py-2.5 rounded-lg bg-sky-500 text-white text-sm hover:bg-sky-600 disabled:opacity-40 transition"
              >
                Continue
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Settings */}
        {step === 3 && (
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-1">Configure generation</h2>
            <p className="text-sm text-gray-500 mb-6">Set the number of scenarios and camera views</p>

            <div className="space-y-6 mb-8">
              {/* Mode */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Scenario mode</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setMode("random")}
                    className={`p-4 rounded-xl border-2 text-left transition ${
                      mode === "random" ? "border-sky-300 bg-sky-50/50" : "border-gray-100 hover:border-gray-200"
                    }`}
                  >
                    <p className={`text-sm font-medium ${mode === "random" ? "text-gray-900" : "text-gray-600"}`}>Random</p>
                    <p className="text-xs text-gray-500 mt-0.5">Procedurally generated with target counts</p>
                  </button>
                  <button
                    onClick={() => setMode("curated")}
                    className={`p-4 rounded-xl border-2 text-left transition ${
                      mode === "curated" ? "border-sky-300 bg-sky-50/50" : "border-gray-100 hover:border-gray-200"
                    }`}
                  >
                    <p className={`text-sm font-medium ${mode === "curated" ? "text-gray-900" : "text-gray-600"}`}>Curated</p>
                    <p className="text-xs text-gray-500 mt-0.5">10 hand-picked scenarios</p>
                  </button>
                </div>
              </div>

              {/* Counts */}
              {mode === "random" && (
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      {environment === "collision_prediction" ? "Hit scenarios" : "Stable scenarios"}
                    </label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range" min={0} max={25} value={numStable}
                        onChange={(e) => setNumStable(Number(e.target.value))}
                        className="flex-1 accent-emerald-500"
                      />
                      <span className="w-8 text-center text-sm font-medium text-emerald-600">{numStable}</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      {environment === "collision_prediction" ? "Miss scenarios" : "Unstable scenarios"}
                    </label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range" min={0} max={25} value={numUnstable}
                        onChange={(e) => setNumUnstable(Number(e.target.value))}
                        className="flex-1 accent-rose-500"
                      />
                      <span className="w-8 text-center text-sm font-medium text-rose-600">{numUnstable}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Views */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Camera views per scenario</label>
                <div className="flex items-center gap-3">
                  <input
                    type="range" min={1} max={6} value={numViews}
                    onChange={(e) => setNumViews(Number(e.target.value))}
                    className="flex-1 accent-sky-500"
                  />
                  <span className="w-8 text-center text-sm font-medium text-gray-900">{numViews}</span>
                </div>
              </div>

              {/* Question Generation */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Question Generation</label>
                <p className="text-xs text-gray-400 mb-3">How should stacking questions be generated?</p>
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <button
                    onClick={() => setQuestionSource("template")}
                    className={`p-3 rounded-xl border-2 text-left transition ${
                      questionSource === "template" ? "border-sky-300 bg-sky-50/50" : "border-gray-100 hover:border-gray-200"
                    }`}
                  >
                    <p className={`text-sm font-medium ${questionSource === "template" ? "text-gray-900" : "text-gray-600"}`}>Template</p>
                    <p className="text-xs text-gray-500 mt-0.5">Auto-generated from scenario</p>
                  </button>
                  <button
                    onClick={() => setQuestionSource("azure")}
                    className={`p-3 rounded-xl border-2 text-left transition ${
                      questionSource === "azure" ? "border-sky-300 bg-sky-50/50" : "border-gray-100 hover:border-gray-200"
                    }`}
                  >
                    <p className={`text-sm font-medium ${questionSource === "azure" ? "text-gray-900" : "text-gray-600"}`}>Azure OpenAI</p>
                    <p className="text-xs text-gray-500 mt-0.5">GPT-4o for richer questions</p>
                  </button>
                </div>
                {questionSource === "azure" && (
                  <p className="text-xs text-sky-600 mt-1">Azure credentials needed below</p>
                )}
              </div>

              {/* Azure OpenAI Credentials — shown when GPT-4o answer model or Azure questions */}
              {(answerModel === "gpt-4o" || questionSource === "azure") && (
                <div className="space-y-3 p-4 rounded-xl bg-gray-50 border border-gray-100">
                  <p className="text-xs font-medium text-gray-700">
                    Azure OpenAI Credentials
                    <span className="text-gray-400 font-normal ml-1">
                      — used for {[answerModel === "gpt-4o" && "GPT-4o answers", questionSource === "azure" && "question generation"].filter(Boolean).join(" + ")}
                    </span>
                  </p>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Endpoint</label>
                    <input
                      type="text"
                      value={azureEndpoint}
                      onChange={(e) => setAzureEndpoint(e.target.value)}
                      placeholder="https://your-resource.openai.azure.com/openai/v1"
                      className="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-sky-400 outline-none bg-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">API Key</label>
                    <input
                      type="password"
                      value={azureKey}
                      onChange={(e) => setAzureKey(e.target.value)}
                      placeholder="Paste your Azure OpenAI key"
                      className="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-sky-400 outline-none bg-white font-mono"
                    />
                  </div>
                  <p className="text-xs text-gray-400">Key is sent to backend only for this job — not stored</p>
                </div>
              )}
            </div>

            {/* Summary */}
            <div className="rounded-xl border border-gray-100 p-5 mb-8">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-2xl font-semibold text-gray-900">{total}</p>
                  <p className="text-xs text-gray-500">scenarios</p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-gray-900">{numViews}</p>
                  <p className="text-xs text-gray-500">views each</p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-gray-900">~{Math.ceil(total * 0.5)}m</p>
                  <p className="text-xs text-gray-500">est. time</p>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <button onClick={() => setStep(2)} className="px-5 py-2.5 rounded-lg border border-gray-200 text-gray-700 text-sm hover:bg-gray-50">
                Back
              </button>
              <button
                onClick={() => setStep(4)}
                className="px-5 py-2.5 rounded-lg bg-sky-500 text-white text-sm hover:bg-sky-600 transition"
              >
                Continue
              </button>
            </div>
          </div>
        )}

        {/* Step 4: Name & Create */}
        {step === 4 && (
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-1">Name your dataset</h2>
            <p className="text-sm text-gray-500 mb-6">Give this evaluation dataset a name for easy reference</p>

            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="e.g. stacking-v1-balanced"
              className="w-full px-4 py-3 rounded-xl border border-gray-200 text-sm focus:border-sky-400 focus:ring-1 focus:ring-sky-400 outline-none mb-1"
            />
            {!datasetName.trim() && (
              <p className="text-xs text-amber-600 mb-3">Dataset name is required</p>
            )}

            {/* Final summary */}
            <div className="rounded-xl bg-gray-50 p-5 mb-8 text-sm text-gray-600 space-y-2">
              <div className="flex justify-between">
                <span>Environment</span>
                <span className="text-gray-900">{environment === "collision_prediction" ? "Collision Prediction" : "Stacking Stability"}</span>
              </div>
              <div className="flex justify-between">
                <span>Mode</span>
                <span className="text-gray-900">{mode === "curated" ? "Curated (10)" : `Random (${numStable} ${environment === "collision_prediction" ? "hit" : "stable"}, ${numUnstable} ${environment === "collision_prediction" ? "miss" : "unstable"})`}</span>
              </div>
              <div className="flex justify-between">
                <span>Objects</span>
                <span className="text-gray-900">{selectedObjects.length} types</span>
              </div>
              <div className="flex justify-between">
                <span>Views</span>
                <span className="text-gray-900">{numViews} per scenario</span>
              </div>
              <div className="flex justify-between">
                <span>Questions</span>
                <span className="text-gray-900">{questionSource === "azure" ? "Azure OpenAI (GPT-4o)" : "Template-based"}</span>
              </div>
              <div className="flex justify-between">
                <span>Total scenarios</span>
                <span className="text-gray-900 font-medium">{total}</span>
              </div>
            </div>

            <div className="flex gap-3">
              <button onClick={() => setStep(3)} className="px-5 py-2.5 rounded-lg border border-gray-200 text-gray-700 text-sm hover:bg-gray-50">
                Back
              </button>
              <button
                onClick={handleCreate}
                disabled={submitting || !datasetName.trim()}
                className={`px-6 py-2.5 rounded-lg text-white text-sm font-medium transition-all ${
                  submitting
                    ? "bg-sky-400 cursor-wait"
                    : "bg-sky-500 hover:bg-sky-600 active:scale-95"
                } disabled:opacity-70`}
              >
                {submitting ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Processing...
                  </span>
                ) : "Create Ground Truths"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
