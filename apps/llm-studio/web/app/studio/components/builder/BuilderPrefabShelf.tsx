import { useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import type { BlockComponent } from "../../../../lib/defaults";
import type { StudioComponent, StudioComponentPrefab } from "../../types";
import { labelForComponentKind } from "../../utils/document";
import { BuilderPrefabEditorPopover } from "./BuilderPrefabEditorPopover";

type BuilderPrefabShelfProps = {
  componentPrefabs: StudioComponentPrefab[];
  updateComponentPrefab: (
    prefabId: string,
    nextName: string,
    nextComponent: StudioComponent
  ) => string | null;
  replaceAllComponentsWithComponentSettings: (
    prefabName: string,
    kind: StudioComponent["kind"],
    component: BlockComponent
  ) => void;
  deleteComponentPrefab: (prefabId: string) => void;
};

type OpenPrefabEditor = {
  prefabId: string;
  anchorEl: HTMLButtonElement;
};

export function BuilderPrefabShelf({
  componentPrefabs,
  updateComponentPrefab,
  replaceAllComponentsWithComponentSettings,
  deleteComponentPrefab,
}: BuilderPrefabShelfProps) {
  const [openEditor, setOpenEditor] = useState<OpenPrefabEditor | null>(null);
  const [editorPosition, setEditorPosition] = useState<{ left: number; top: number } | null>(null);
  const editorRef = useRef<HTMLDivElement | null>(null);

  const openPrefab = useMemo(
    () => componentPrefabs.find((prefab) => prefab.id === openEditor?.prefabId) ?? null,
    [componentPrefabs, openEditor?.prefabId]
  );

  useEffect(() => {
    if (!openEditor) {
      return;
    }

    function handlePointerDown(event: PointerEvent): void {
      const target = event.target;
      if (target instanceof Element && target.closest("[data-prefab-editor-root]")) {
        return;
      }
      setOpenEditor(null);
      setEditorPosition(null);
    }

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        setOpenEditor(null);
        setEditorPosition(null);
      }
    }

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [openEditor]);

  useEffect(() => {
    if (!openEditor) {
      return;
    }
    const stillExists = componentPrefabs.some((prefab) => prefab.id === openEditor.prefabId);
    if (!stillExists) {
      setOpenEditor(null);
      setEditorPosition(null);
    }
  }, [componentPrefabs, openEditor]);

  useLayoutEffect(() => {
    if (!openEditor || typeof window === "undefined") {
      setEditorPosition(null);
      return;
    }

    let rafId = 0;

    const updatePosition = () => {
      const anchorEl = openEditor.anchorEl;
      if (!anchorEl.isConnected || !document.body.contains(anchorEl)) {
        setOpenEditor(null);
        setEditorPosition(null);
        return;
      }

      const anchorRect = anchorEl.getBoundingClientRect();
      const popoverRect = editorRef.current?.getBoundingClientRect();
      const popoverWidth = popoverRect?.width ?? 372;
      const popoverHeight = popoverRect?.height ?? 520;

      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const margin = 8;
      const gap = 8;

      let left = anchorRect.left;
      if (left + popoverWidth > viewportWidth - margin) {
        left = anchorRect.right - popoverWidth;
      }

      const belowTop = anchorRect.bottom + gap;
      const aboveTop = anchorRect.top - popoverHeight - gap;
      const top =
        belowTop + popoverHeight <= viewportHeight - margin || aboveTop < margin
          ? belowTop
          : aboveTop;

      left = Math.min(Math.max(left, margin), viewportWidth - popoverWidth - margin);
      const boundedTop = Math.min(Math.max(top, margin), viewportHeight - popoverHeight - margin);

      setEditorPosition((current) =>
        current && current.left === left && current.top === boundedTop
          ? current
          : { left, top: boundedTop }
      );
    };

    const scheduleUpdate = () => {
      window.cancelAnimationFrame(rafId);
      rafId = window.requestAnimationFrame(updatePosition);
    };

    scheduleUpdate();
    window.requestAnimationFrame(scheduleUpdate);

    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("scroll", scheduleUpdate, true);
    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", scheduleUpdate);
      window.removeEventListener("scroll", scheduleUpdate, true);
    };
  }, [openEditor]);

  function openEditorFromEvent(event: MouseEvent<HTMLButtonElement>, prefabId: string): void {
    setOpenEditor((current) =>
      current?.prefabId === prefabId
        ? null
        : {
            prefabId,
            anchorEl: event.currentTarget,
          }
    );
  }

  return (
    <section className="prefabShelf" aria-label="Component prefabs">
      <div className="prefabShelfHead compact">
        <h3>Prefabs</h3>
        <span className="prefabShelfCount">{componentPrefabs.length}</span>
      </div>

      {componentPrefabs.length === 0 ? (
        <p className="prefabShelfEmpty">No prefabs yet. Save one from a component card.</p>
      ) : (
        <div className="prefabChipList" role="list" aria-label="Saved component prefabs">
          {componentPrefabs.map((prefab) => {
            const isOpen = openEditor?.prefabId === prefab.id;
            return (
              <button
                key={prefab.id}
                type="button"
                className={`prefabChipButton${isOpen ? " isActive" : ""}`}
                onClick={(event) => openEditorFromEvent(event, prefab.id)}
                title={`Edit ${prefab.name}`}
                aria-expanded={isOpen}
                aria-haspopup="dialog"
                role="listitem"
              >
                <span className={`prefabChipKind kind-${prefab.kind}`}>{labelForComponentKind(prefab.kind)}</span>
                <span className="prefabChipName">{prefab.name}</span>
              </button>
            );
          })}
        </div>
      )}

      <BuilderPrefabEditorPopover
        prefab={openPrefab}
        popoverRef={editorRef}
        position={editorPosition}
        closeEditor={() => {
          setOpenEditor(null);
          setEditorPosition(null);
        }}
        updateComponentPrefab={updateComponentPrefab}
        replaceAllComponentsWithComponentSettings={replaceAllComponentsWithComponentSettings}
        deleteComponentPrefab={deleteComponentPrefab}
      />
    </section>
  );
}
