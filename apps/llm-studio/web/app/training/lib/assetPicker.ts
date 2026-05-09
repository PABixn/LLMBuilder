import type { ProjectSummary } from "../../../lib/api";
import type { TrainingJob as TokenizerTrainingJob } from "../../../lib/tokenizerLegacyApi";
import { asString } from "./object";

export function normalizeAssetPickerQuery(query: string): string {
  return query.trim().toLowerCase();
}

export function filterPickerProjects(
  projects: ProjectSummary[],
  query: string
): ProjectSummary[] {
  const normalizedQuery = normalizeAssetPickerQuery(query);
  return [...projects]
    .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
    .filter((project) => {
      if (normalizedQuery === "") {
        return true;
      }
      return [project.name, project.id, project.artifact_file, project.artifact_path].some(
        (value) =>
          typeof value === "string" && value.toLowerCase().includes(normalizedQuery)
      );
    });
}

export function filterPickerTokenizerJobs(
  jobs: TokenizerTrainingJob[],
  query: string
): TokenizerTrainingJob[] {
  const normalizedQuery = normalizeAssetPickerQuery(query);
  return [...jobs]
    .filter((job) => job.status === "completed")
    .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
    .filter((job) => {
      if (normalizedQuery === "") {
        return true;
      }
      const tokenizerName = asString(job.tokenizer_config.name, job.id);
      return [tokenizerName, job.id, job.artifact_file, job.artifact_path].some(
        (value) =>
          typeof value === "string" && value.toLowerCase().includes(normalizedQuery)
      );
    });
}
