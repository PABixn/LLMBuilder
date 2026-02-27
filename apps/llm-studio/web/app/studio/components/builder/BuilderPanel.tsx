import { FiChevronDown, FiChevronRight, FiCornerUpLeft, FiCornerUpRight, FiPlus } from "react-icons/fi";

import { BuilderCanvasContent } from "./BuilderCanvasContent";
import { BuilderPrefabShelf } from "./BuilderPrefabShelf";
import { InsertMenuPortal } from "./InsertMenuPortal";
import type { BuilderPanelProps } from "./types";
import { useBuilderInsertMenu } from "./useBuilderInsertMenu";

export type { BuilderPanelProps } from "./types";

export function BuilderPanel(props: BuilderPanelProps) {
  const {
    openInsertMenu,
    insertMenuPosition,
    insertMenuRef,
    closeInsertMenu,
    openInsertMenuFromEvent,
    handleToggleKeyDown,
  } = useBuilderInsertMenu();

  const {
    addBlock,
    canRedoDocument,
    canUndoDocument,
    collapseAllCanvasNodes,
    componentPrefabs,
    deleteComponentPrefab,
    expandAllCanvasNodes,
    metrics,
    redoDocument,
    replaceAllComponentsWithComponentSettings,
    undoDocument,
    updateComponentPrefab,
  } = props;

  return (
    <>
      <section id="block-builder" className="panelCard builderPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Visual Builder</p>
            <h2>Visual designer</h2>
            <p className="panelCopy">
              Build depth horizontally. Use insert slots to add blocks/components/MLP steps, then drag to reorder.
            </p>
          </div>
          <div className="actionCluster">
            <button
              type="button"
              className="buttonGhost"
              onClick={undoDocument}
              disabled={!canUndoDocument}
              title="Undo (Ctrl/Cmd+Z)"
            >
              <FiCornerUpLeft /> Undo
            </button>
            <button
              type="button"
              className="buttonGhost"
              onClick={redoDocument}
              disabled={!canRedoDocument}
              title="Redo (Ctrl/Cmd+Shift+Z or Ctrl+Y)"
            >
              <FiCornerUpRight /> Redo
            </button>
            <button type="button" className="buttonGhost" onClick={collapseAllCanvasNodes}>
              <FiChevronRight /> Collapse all
            </button>
            <button type="button" className="buttonGhost" onClick={expandAllCanvasNodes}>
              <FiChevronDown /> Expand all
            </button>
            <button type="button" className="buttonGhost" onClick={addBlock}>
              <FiPlus /> Add block
            </button>
          </div>
        </div>

        <div className="builderCanvasToolbar">
          <div className="builderCanvasHint">
            <span className="builderCanvasHintDot" aria-hidden />
            Two-axis canvas: blocks are columns, components stay nested inside each block.
          </div>
          <div className="builderCanvasStats" aria-label="Builder canvas statistics">
            <span>{metrics.blockCount} blocks</span>
            <span>{metrics.componentCount} components</span>
            <span>{metrics.mlpStepCount} MLP steps</span>
          </div>
        </div>

        <BuilderPrefabShelf
          componentPrefabs={componentPrefabs}
          updateComponentPrefab={updateComponentPrefab}
          deleteComponentPrefab={deleteComponentPrefab}
          replaceAllComponentsWithComponentSettings={replaceAllComponentsWithComponentSettings}
        />

        <BuilderCanvasContent
          {...props}
          openInsertMenu={openInsertMenu}
          openInsertMenuFromEvent={openInsertMenuFromEvent}
          handleToggleKeyDown={handleToggleKeyDown}
        />
      </section>

      <InsertMenuPortal
        openInsertMenu={openInsertMenu}
        insertMenuPosition={insertMenuPosition}
        insertMenuRef={insertMenuRef}
        closeInsertMenu={closeInsertMenu}
      />
    </>
  );
}
