// Phase 7 placeholders — fleshed out with live API calls in Phase 8.
export function ComingSoon({ title, blurb, phase }: { title: string; blurb: string; phase: number }) {
  return (
    <div className="space-y-6">
      <header>
        <div className="eyebrow">Phase {phase}</div>
        <h1 className="text-3xl mt-1">{title}</h1>
        <p className="text-muted mt-2 max-w-2xl">{blurb}</p>
      </header>
      <div className="card p-10 text-center text-faint">
        <div className="eyebrow">Wired in Phase {phase}</div>
        <p className="mt-2 text-sm">Backend route is live; the interactive screen lands in Phase 8.</p>
      </div>
    </div>
  );
}

export const Models = () => <ComingSoon phase={8} title="Models & MoE merge/split"
  blurb="Download LLaDA2.0-mini/flash, merge separate→merged experts, detect format, and manage the local model inventory." />;
export const Datasets = () => <ComingSoon phase={8} title="Datasets & preparation"
  blurb="Ingest, map columns to messages, validate, preview, and build the GSM8K conversational preset." />;
export const Training = () => <ComingSoon phase={8} title="Training — block-diffusion SFT"
  blurb="Configure every YAML key (incl. noise_range, block_size), launch torchrun, watch live metrics, resume." />;
export const Export = () => <ComingSoon phase={8} title="Export & packaging"
  blurb="Auto-split the checkpoint, copy the modeling file, verify completeness, and package a runnable model." />;
export const Chat = () => <ComingSoon phase={8} title="Chat & diffusion playground"
  blurb="Run inference with dLLM params (denoising steps, masking schedule), compare base vs fine-tuned." />;
export const Settings = () => <ComingSoon phase={8} title="Settings"
  blurb="Backend bind, HuggingFace token, data dir, GPU selection, theme." />;
