"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { SETTINGS_CATEGORY_HASH_MAP } from "../constants";
import type { SettingsCategory } from "../types";

export function useTokenizerSettingsNavigation() {
  const [highlightedSettingsCategory, setHighlightedSettingsCategory] =
    useState<SettingsCategory | null>(null);
  const settingsCategoryHighlightTimeoutRef = useRef<number | null>(null);
  const tokenizerAndTrainingPanelRef = useRef<HTMLDetailsElement | null>(null);
  const datasetPanelRef = useRef<HTMLDetailsElement | null>(null);
  const tokenizerCategoryRef = useRef<HTMLDivElement | null>(null);
  const datasetCategoryRef = useRef<HTMLDivElement | null>(null);
  const trainingCategoryRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    return () => {
      if (settingsCategoryHighlightTimeoutRef.current !== null) {
        window.clearTimeout(settingsCategoryHighlightTimeoutRef.current);
      }
    };
  }, []);

  const openSettingsCategory = useCallback((category: SettingsCategory) => {
    if (tokenizerAndTrainingPanelRef.current) {
      tokenizerAndTrainingPanelRef.current.open = true;
    }
    if (datasetPanelRef.current) {
      datasetPanelRef.current.open = true;
    }

    const targetRef =
      category === "tokenizer"
        ? tokenizerCategoryRef
        : category === "dataset"
          ? datasetCategoryRef
          : trainingCategoryRef;

    const hash = SETTINGS_CATEGORY_HASH_MAP[category];
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }

    targetRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
      inline: "nearest",
    });

    setHighlightedSettingsCategory(category);
    if (settingsCategoryHighlightTimeoutRef.current !== null) {
      window.clearTimeout(settingsCategoryHighlightTimeoutRef.current);
    }
    settingsCategoryHighlightTimeoutRef.current = window.setTimeout(() => {
      setHighlightedSettingsCategory((previous) =>
        previous === category ? null : previous
      );
    }, 1800);
  }, []);

  const handleSettingsCategoryNavigation = useCallback(
    (category: SettingsCategory) => {
      openSettingsCategory(category);
    },
    [openSettingsCategory]
  );

  useEffect(() => {
    const hash = window.location.hash;
    if (hash === SETTINGS_CATEGORY_HASH_MAP.tokenizer) {
      openSettingsCategory("tokenizer");
      return;
    }
    if (hash === SETTINGS_CATEGORY_HASH_MAP.dataset) {
      openSettingsCategory("dataset");
      return;
    }
    if (hash === SETTINGS_CATEGORY_HASH_MAP.training) {
      openSettingsCategory("training");
    }
  }, [openSettingsCategory]);

  return {
    highlightedSettingsCategory,
    handleSettingsCategoryNavigation,
    tokenizerAndTrainingPanelRef,
    datasetPanelRef,
    tokenizerCategoryRef,
    datasetCategoryRef,
    trainingCategoryRef,
  };
}
