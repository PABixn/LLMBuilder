import { FiChevronDown, FiChevronRight, FiCornerUpLeft, FiCornerUpRight, FiPlus } from "react-icons/fi";

import { BuilderCanvasContent } from "./BuilderCanvasContent";
import { BuilderPrefabShelf } from "./BuilderPrefabShelf";
import { InsertMenuPortal } from "./InsertMenuPortal";
import type { BuilderPanelProps } from "./types";
import { useBuilderInsertMenu } from "./useBuilderInsertMenu";
import { HelpTooltip, InfoTooltip } from "../../../shared/components/HelpTooltip";

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
            <p className="panelEyebrow">Builder</p>
            <h2>
              Visual designer
              <InfoTooltip label="Visual designer explanation" align="left" width="wide">
                <p>
                  The canvas edits the architecture structure directly. Blocks are ordered
                  transformer layers; each block contains attention, MLP, norm, activation,
                  or linear components.
                </p>
              </InfoTooltip>
            </h2>
            <p className="panelCopy">
              Add blocks, components, and MLP steps. Drag to reorder.
            </p>
          </div>
          <div className="actionCluster">
            <HelpTooltip label="Undo" content="Reverts the last builder edit. Keyboard shortcut: Ctrl/Cmd+Z.">
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={undoDocument}
                disabled={!canUndoDocument}
                aria-label="Undo"
              >
                <FiCornerUpLeft />
              </button>
            </HelpTooltip>
            <HelpTooltip label="Redo" content="Reapplies the last undone builder edit. Keyboard shortcut: Ctrl/Cmd+Shift+Z or Ctrl+Y.">
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={redoDocument}
                disabled={!canRedoDocument}
                aria-label="Redo"
              >
                <FiCornerUpRight />
              </button>
            </HelpTooltip>
            <HelpTooltip label="Collapse all" content="Closes all block and component detail panels so the canvas is easier to scan.">
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={collapseAllCanvasNodes}
                aria-label="Collapse all"
              >
                <FiChevronRight />
              </button>
            </HelpTooltip>
            <HelpTooltip label="Expand all" content="Opens all block and component panels so every editable setting is visible.">
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={expandAllCanvasNodes}
                aria-label="Expand all"
              >
                <FiChevronDown />
              </button>
            </HelpTooltip>
            <HelpTooltip label="Add block" content="Adds another transformer block at the end of the model. More blocks increase depth and parameter count.">
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={addBlock}
                aria-label="Add block"
              >
                <FiPlus />
              </button>
            </HelpTooltip>
          </div>
        </div>

        <div className="builderCanvasToolbar">
          <div className="builderCanvasHint">
            <span className="builderCanvasHintDot" aria-hidden />
            Blocks are columns. Components stay inside each block.
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
