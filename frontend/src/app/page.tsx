"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

type Counts = { datasets: number; evalRuns: number; projects: number };

const STAGES = [
  {
    num: 1, title: "Generate Ground Truths", href: "/generate", color: "sky",
    desc: "Create physics-verified scenarios with MuJoCo simulation",
    sub: "No model needed — pure ground truth",
    icon: <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5m8.25 3v6.75m0 0l-3-3m3 3l3-3M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z" /></svg>,
  },
  {
    num: 2, title: "Evaluate Model", href: "/evaluate", color: "violet",
    desc: "Run any VLM against ground truth and measure accuracy",
    sub: "GPT-4o, Qwen, or any model",
    icon: <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" /></svg>,
  },
  {
    num: 3, title: "Create RLHF Data", href: "/rlhf", color: "emerald",
    desc: "Pair preferred vs rejected outputs into DPO training data",
    sub: "Annotate, review, and export",
    icon: <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" /></svg>,
  },
];

const cm: Record<string, { bg: string; border: string; text: string; badge: string; arrow: string }> = {
  sky:     { bg: "bg-sky-50", border: "border-sky-200 hover:border-sky-400 hover:shadow-md", text: "text-sky-600", badge: "bg-sky-500", arrow: "group-hover:translate-x-1" },
  violet:  { bg: "bg-violet-50", border: "border-violet-200 hover:border-violet-400 hover:shadow-md", text: "text-violet-600", badge: "bg-violet-500", arrow: "group-hover:translate-x-1" },
  emerald: { bg: "bg-emerald-50", border: "border-emerald-200 hover:border-emerald-400 hover:shadow-md", text: "text-emerald-600", badge: "bg-emerald-500", arrow: "group-hover:translate-x-1" },
};

export default function Home() {
  const [counts, setCounts] = useState<Counts>({ datasets: 0, evalRuns: 0, projects: 0 });

  useEffect(() => {
    Promise.all([
      fetch("/api/datasets").then(r => r.json()).then(d => d.length).catch(() => 0),
      fetch("/api/eval-runs").then(r => r.json()).then(d => d.length).catch(() => 0),
      fetch("/api/projects").then(r => r.json()).then(d => d.length).catch(() => 0),
    ]).then(([datasets, evalRuns, projects]) => setCounts({ datasets, evalRuns, projects }));
  }, []);

  const badges = [counts.datasets + " datasets", counts.evalRuns + " evaluations", counts.projects + " projects"];

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
            <Link href="/" className="text-sky-600 font-medium">Home</Link>
            <Link href="/datasets" className="hover:text-gray-900">Datasets</Link>
            <Link href="/generate" className="hover:text-gray-900">Generate</Link>
            <Link href="/evaluate" className="hover:text-gray-900">Evaluate</Link>
            <Link href="/rlhf" className="hover:text-gray-900">RLHF</Link>
          </div>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto px-8 py-16">
        <div className="text-center mb-14">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">Spatial Reasoning RLHF Pipeline</h1>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto">
            Generate physics-verified ground truths, evaluate vision-language models, and create preference data for training.
          </p>
          {(counts.datasets > 0 || counts.evalRuns > 0 || counts.projects > 0) && (
            <div className="flex items-center justify-center gap-3 mt-5">
              {badges.map((b, i) => (
                <span key={i} className="text-xs bg-gray-100 text-gray-500 px-3 py-1 rounded-full">{b}</span>
              ))}
            </div>
          )}
        </div>

        {/* Pipeline flow */}
        <div className="grid grid-cols-3 gap-6">
          {STAGES.map((s, i) => {
            const c = cm[s.color];
            return (
              <div key={s.num} className="relative">
                {i < 2 && (
                  <div className="hidden lg:block absolute top-1/2 -right-3 z-10 text-gray-300">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" /></svg>
                  </div>
                )}
                <Link href={s.href} className={`group block rounded-2xl border p-6 transition-all ${c.border}`}>
                  <div className="flex items-center gap-3 mb-4">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${c.bg} ${c.text}`}>
                      {s.icon}
                    </div>
                    <span className={`text-[11px] font-bold px-2.5 py-1 rounded-full text-white ${c.badge}`}>
                      Stage {s.num}
                    </span>
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-1.5">{s.title}</h3>
                  <p className="text-sm text-gray-500 mb-1">{s.desc}</p>
                  <p className="text-xs text-gray-400">{s.sub}</p>
                  <div className={`mt-4 flex items-center gap-1 text-sm font-medium ${c.text}`}>
                    Get started
                    <svg className={`w-4 h-4 transition-transform ${c.arrow}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" /></svg>
                  </div>
                </Link>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
