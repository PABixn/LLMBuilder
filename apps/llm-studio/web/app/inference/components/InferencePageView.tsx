import { AppTopNav } from "../../shared/components/AppTopNav";
import type { useInferenceController } from "../hooks/useInferenceController";
import { InferenceCheckpointPickerDialog } from "./InferenceCheckpointPickerDialog";
import { InferenceComposer } from "./InferenceComposer";
import { InferenceModelPickerDialog } from "./InferenceModelPickerDialog";
import { InferenceOutputPanel } from "./InferenceOutputPanel";
import { TrainingArtifactPanel } from "./TrainingArtifactPanel";

type InferenceController = ReturnType<typeof useInferenceController>;

type InferencePageViewProps = {
  controller: InferenceController;
};

export function InferencePageView({ controller }: InferencePageViewProps) {
  const {
    error,
    loading,
    generating,
    selectedJobId,
    setSelectedJobId,
    selectedJob,
    completedJobs,
    prompt,
    setPrompt,
    maxTokens,
    setMaxTokens,
    temperature,
    setTemperature,
    topK,
    setTopK,
    seed,
    setSeed,
    repetitionPenalty,
    setRepetitionPenalty,
    result,
    pickerOpen,
    setPickerOpen,
    pickerQuery,
    setPickerQuery,
    visiblePickerJobs,
    checkpoints,
    checkpointValue,
    setCheckpointValue,
    checkpointPickerOpen,
    setCheckpointPickerOpen,
    checkpointPickerQuery,
    setCheckpointPickerQuery,
    checkpointsLoading,
    checkpointError,
    latestCheckpoint,
    selectedCheckpoint,
    visibleCheckpointOptions,
    showLatestCheckpointOption,
    closePicker,
    closeCheckpointPicker,
    refreshJobs,
    refreshCheckpoints,
    handleGenerate,
    pickerSearchPlaceholder,
    checkpointSearchPlaceholder,
    latestCheckpointValue,
  } = controller;

  return (
    <main className="studioRoot inferencePage">
      <AppTopNav />

      {error ? <div className="inlineNotice tone-error">Inference problem: {error}</div> : null}

      <section className="inferenceLayout">
        <TrainingArtifactPanel
          selectedJob={selectedJob}
          loading={loading}
          pickerOpen={pickerOpen}
          checkpointPickerOpen={checkpointPickerOpen}
          checkpointValue={checkpointValue}
          latestCheckpointValue={latestCheckpointValue}
          selectedCheckpoint={selectedCheckpoint}
          checkpointsLoading={checkpointsLoading}
          checkpointError={checkpointError}
          onRefreshJobs={() => void refreshJobs()}
          onOpenModelPicker={() => setPickerOpen(true)}
          onOpenCheckpointPicker={() => setCheckpointPickerOpen(true)}
        />

        <InferenceComposer
          prompt={prompt}
          maxTokens={maxTokens}
          temperature={temperature}
          topK={topK}
          seed={seed}
          repetitionPenalty={repetitionPenalty}
          generating={generating}
          canGenerate={Boolean(selectedJob) && prompt.trim() !== "" && !generating}
          onPromptChange={setPrompt}
          onMaxTokensChange={setMaxTokens}
          onTemperatureChange={setTemperature}
          onTopKChange={setTopK}
          onSeedChange={setSeed}
          onRepetitionPenaltyChange={setRepetitionPenalty}
          onGenerate={handleGenerate}
        />
      </section>

      <InferenceOutputPanel result={result} />

      <InferenceModelPickerDialog
        open={pickerOpen}
        loading={loading}
        error={error}
        selectedJobId={selectedJobId}
        completedJobs={completedJobs}
        visibleJobs={visiblePickerJobs}
        query={pickerQuery}
        searchPlaceholder={pickerSearchPlaceholder}
        onClose={closePicker}
        onQueryChange={setPickerQuery}
        onRefresh={() => void refreshJobs()}
        onClearSearch={() => setPickerQuery("")}
        onSelectJob={(jobId) => {
          setSelectedJobId(jobId);
          closePicker();
        }}
      />

      <InferenceCheckpointPickerDialog
        open={checkpointPickerOpen}
        selectedJob={selectedJob}
        checkpointsLoading={checkpointsLoading}
        checkpointError={checkpointError}
        checkpoints={checkpoints}
        latestCheckpoint={latestCheckpoint}
        visibleCheckpointOptions={visibleCheckpointOptions}
        checkpointValue={checkpointValue}
        latestCheckpointValue={latestCheckpointValue}
        showLatestCheckpointOption={showLatestCheckpointOption}
        query={checkpointPickerQuery}
        searchPlaceholder={checkpointSearchPlaceholder}
        onClose={closeCheckpointPicker}
        onQueryChange={setCheckpointPickerQuery}
        onRefresh={() => {
          if (!selectedJob) {
            return;
          }
          void refreshCheckpoints(selectedJob.id);
        }}
        onClearSearch={() => setCheckpointPickerQuery("")}
        onSelectCheckpoint={(value) => {
          setCheckpointValue(value);
          closeCheckpointPicker();
        }}
      />
    </main>
  );
}
