"use client";

import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";

import {
  ACTIVE_JOB_STORAGE_KEY,
  DATASET_FORM_STORAGE_KEY,
  HIDDEN_RECENT_JOB_IDS_STORAGE_KEY,
  PREVIEW_TEXT_STORAGE_KEY,
  TOKENIZER_FORM_STORAGE_KEY,
  TRAINING_FORM_STORAGE_KEY,
} from "../constants";
import {
  hydrateDatasetForm,
  hydrateTrainingForm,
  hydrateTokenizerForm,
} from "../lib/dataset";
import { hydratePreviewText } from "../lib/preview";
import {
  readStoredJson,
  readStoredStringArray,
  readStoredValue,
  removeStoredValue,
  writeStoredJson,
  writeStoredValue,
} from "../lib/storage";
import type { DatasetFormState, TokenizerFormState, TrainingFormState } from "../types";

type UseTokenizerPersistenceArgs = {
  tokenizerForm: TokenizerFormState;
  setTokenizerForm: Dispatch<SetStateAction<TokenizerFormState>>;
  datasetForm: DatasetFormState;
  setDatasetForm: Dispatch<SetStateAction<DatasetFormState>>;
  trainingForm: TrainingFormState;
  setTrainingForm: Dispatch<SetStateAction<TrainingFormState>>;
  previewText: string;
  setPreviewText: Dispatch<SetStateAction<string>>;
  activeJobId: string | null;
  setActiveJobId: Dispatch<SetStateAction<string | null>>;
  hiddenRecentJobIds: string[];
  setHiddenRecentJobIds: Dispatch<SetStateAction<string[]>>;
};

type TokenizerPersistenceState = {
  hasHydratedLocalState: boolean;
  hasHydratedLocalStateRef: MutableRefObject<boolean>;
};

export function useTokenizerPersistence({
  tokenizerForm,
  setTokenizerForm,
  datasetForm,
  setDatasetForm,
  trainingForm,
  setTrainingForm,
  previewText,
  setPreviewText,
  activeJobId,
  setActiveJobId,
  hiddenRecentJobIds,
  setHiddenRecentJobIds,
}: UseTokenizerPersistenceArgs): TokenizerPersistenceState {
  const [hasHydratedLocalState, setHasHydratedLocalState] = useState(false);
  const hasHydratedLocalStateRef = useRef(false);

  useEffect(() => {
    const storedTokenizerForm = readStoredJson(TOKENIZER_FORM_STORAGE_KEY);
    if (storedTokenizerForm !== null) {
      setTokenizerForm((previous) => hydrateTokenizerForm(storedTokenizerForm, previous));
    }

    const storedDatasetForm = readStoredJson(DATASET_FORM_STORAGE_KEY);
    if (storedDatasetForm !== null) {
      setDatasetForm((previous) => hydrateDatasetForm(storedDatasetForm, previous));
    }

    const storedTrainingForm = readStoredJson(TRAINING_FORM_STORAGE_KEY);
    if (storedTrainingForm !== null) {
      setTrainingForm((previous) => hydrateTrainingForm(storedTrainingForm, previous));
    }

    const storedPreviewText = readStoredValue(PREVIEW_TEXT_STORAGE_KEY);
    if (storedPreviewText !== null) {
      setPreviewText((previous) => hydratePreviewText(storedPreviewText, previous));
    }

    const storedActiveJobId = readStoredValue(ACTIVE_JOB_STORAGE_KEY);
    if (storedActiveJobId !== null && storedActiveJobId.trim() !== "") {
      setActiveJobId(storedActiveJobId.trim());
    }
    setHiddenRecentJobIds(readStoredStringArray(HIDDEN_RECENT_JOB_IDS_STORAGE_KEY));

    hasHydratedLocalStateRef.current = true;
    setHasHydratedLocalState(true);
  }, [
    setActiveJobId,
    setDatasetForm,
    setHiddenRecentJobIds,
    setPreviewText,
    setTokenizerForm,
    setTrainingForm,
  ]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredJson(TOKENIZER_FORM_STORAGE_KEY, tokenizerForm);
  }, [hasHydratedLocalState, tokenizerForm]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredJson(DATASET_FORM_STORAGE_KEY, datasetForm);
  }, [datasetForm, hasHydratedLocalState]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredJson(TRAINING_FORM_STORAGE_KEY, trainingForm);
  }, [hasHydratedLocalState, trainingForm]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredValue(PREVIEW_TEXT_STORAGE_KEY, previewText);
  }, [hasHydratedLocalState, previewText]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    if (activeJobId === null) {
      removeStoredValue(ACTIVE_JOB_STORAGE_KEY);
      return;
    }
    writeStoredValue(ACTIVE_JOB_STORAGE_KEY, activeJobId);
  }, [activeJobId, hasHydratedLocalState]);

  useEffect(() => {
    if (!hasHydratedLocalState) {
      return;
    }
    writeStoredJson(HIDDEN_RECENT_JOB_IDS_STORAGE_KEY, hiddenRecentJobIds);
  }, [hasHydratedLocalState, hiddenRecentJobIds]);

  return {
    hasHydratedLocalState,
    hasHydratedLocalStateRef,
  };
}
