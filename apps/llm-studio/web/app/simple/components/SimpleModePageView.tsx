import { AppTopNav } from "../../shared/components/AppTopNav";
import type { SimpleModeController } from "../types";
import { ArchitectureStep } from "./ArchitectureStep";
import { InferenceStep } from "./InferenceStep";
import { SimpleStepper } from "./SimpleStepper";
import { SimpleStepShell } from "./SimpleStepShell";
import { TokenizerStep } from "./TokenizerStep";
import { TrainingStep } from "./TrainingStep";

type SimpleModePageViewProps = {
  controller: SimpleModeController;
};

export function SimpleModePageView({ controller }: SimpleModePageViewProps) {
  const { activeStep, setActiveStep, steps } = controller;

  return (
    <main className="studioRoot simpleModePage">
      <AppTopNav activeSimpleStep={activeStep} />

      <section className="simpleHeroPanel">
        <div>
          <p className="simpleEyebrow">Simple Mode</p>
          <h1>Build, train, and test</h1>
          <p>
            Follow the guided path and keep every artifact compatible with Expert Mode.
          </p>
        </div>
        <a className="buttonGhost" href="/">
          Workspace
        </a>
      </section>

      <SimpleStepper
        activeStep={activeStep}
        steps={steps}
        onStepSelect={setActiveStep}
      />

      {steps.map((step) => (
        <SimpleStepShell
          key={step.id}
          activeStep={activeStep}
          step={step}
          onEdit={setActiveStep}
        >
          {step.id === "architecture" ? <ArchitectureStep controller={controller} /> : null}
          {step.id === "tokenizer" ? <TokenizerStep controller={controller} /> : null}
          {step.id === "training" ? <TrainingStep controller={controller} /> : null}
          {step.id === "inference" ? <InferenceStep controller={controller} /> : null}
        </SimpleStepShell>
      ))}
    </main>
  );
}
