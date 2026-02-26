import type { RefObject } from "react";
import { createPortal } from "react-dom";

import type { OpenInsertMenu } from "./types";

type InsertMenuPortalProps = {
  openInsertMenu: OpenInsertMenu | null;
  insertMenuPosition: { left: number; top: number } | null;
  insertMenuRef: RefObject<HTMLDivElement | null>;
  closeInsertMenu: () => void;
};

export function InsertMenuPortal({
  openInsertMenu,
  insertMenuPosition,
  insertMenuRef,
  closeInsertMenu,
}: InsertMenuPortalProps) {
  if (!openInsertMenu || typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div
      ref={insertMenuRef}
      className={`blockInsertMenu insertMenuPortal variant-${openInsertMenu.variant}`}
      data-insert-slot
      role="menu"
      aria-label={openInsertMenu.title}
      style={
        insertMenuPosition
          ? { left: insertMenuPosition.left, top: insertMenuPosition.top }
          : { left: -9999, top: -9999 }
      }
    >
      <div className="blockInsertMenuHeader">{openInsertMenu.title}</div>
      {openInsertMenu.items.map((item) => (
        <button
          key={item.id}
          type="button"
          className="blockInsertMenuButton"
          role="menuitem"
          onClick={() => {
            item.onSelect();
            closeInsertMenu();
          }}
        >
          <span className="blockInsertMenuButtonTitle">{item.label}</span>
        </button>
      ))}
    </div>,
    document.body
  );
}
