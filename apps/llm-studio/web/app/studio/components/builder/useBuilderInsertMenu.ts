import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent,
  type RefObject,
} from "react";

import type { OpenInsertMenu } from "./types";

export interface BuilderInsertMenuController {
  openInsertMenu: OpenInsertMenu | null;
  insertMenuPosition: { left: number; top: number } | null;
  insertMenuRef: RefObject<HTMLDivElement | null>;
  closeInsertMenu: () => void;
  openInsertMenuFromEvent: (
    event: MouseEvent<HTMLButtonElement>,
    config: Omit<OpenInsertMenu, "anchorEl">
  ) => void;
  handleToggleKeyDown: (
    event: ReactKeyboardEvent<HTMLElement>,
    toggle: () => void
  ) => void;
}

export function useBuilderInsertMenu(): BuilderInsertMenuController {
  const [openInsertMenu, setOpenInsertMenu] = useState<OpenInsertMenu | null>(null);
  const [insertMenuPosition, setInsertMenuPosition] = useState<{ left: number; top: number } | null>(
    null
  );
  const insertMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (openInsertMenu === null) {
      return;
    }

    function handlePointerDown(event: PointerEvent): void {
      const target = event.target;
      if (target instanceof Element && target.closest("[data-insert-slot]")) {
        return;
      }
      closeInsertMenu();
    }

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        closeInsertMenu();
      }
    }

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [openInsertMenu]);

  useLayoutEffect(() => {
    if (!openInsertMenu || typeof window === "undefined") {
      setInsertMenuPosition(null);
      return;
    }

    let rafId = 0;

    const updatePosition = () => {
      const anchorEl = openInsertMenu.anchorEl;
      if (!anchorEl.isConnected || !document.body.contains(anchorEl)) {
        setOpenInsertMenu(null);
        setInsertMenuPosition(null);
        return;
      }

      const anchorRect = anchorEl.getBoundingClientRect();
      const menuRect = insertMenuRef.current?.getBoundingClientRect();
      const menuWidth = menuRect?.width ?? (openInsertMenu.variant === "rail" ? 204 : 156);
      const menuHeight = menuRect?.height ?? 180;

      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const gap = 8;
      const margin = 8;

      let left = 0;
      let top = 0;

      if (openInsertMenu.variant === "rail") {
        const preferredRight = anchorRect.right + gap;
        const fallbackLeft = anchorRect.left - menuWidth - gap;
        left =
          preferredRight + menuWidth <= viewportWidth - margin
            ? preferredRight
            : fallbackLeft;
        top = anchorRect.top + 8;
      } else {
        left = anchorRect.left + anchorRect.width / 2 - menuWidth / 2;
        const belowTop = anchorRect.bottom + 6;
        const aboveTop = anchorRect.top - menuHeight - 6;
        top =
          belowTop + menuHeight <= viewportHeight - margin || aboveTop < margin
            ? belowTop
            : aboveTop;
      }

      left = Math.min(Math.max(left, margin), viewportWidth - menuWidth - margin);
      top = Math.min(Math.max(top, margin), viewportHeight - menuHeight - margin);

      setInsertMenuPosition((current) =>
        current && current.left === left && current.top === top
          ? current
          : { left, top }
      );
    };

    const scheduleUpdate = () => {
      window.cancelAnimationFrame(rafId);
      rafId = window.requestAnimationFrame(updatePosition);
    };

    scheduleUpdate();
    // Re-measure after first paint when the menu node has dimensions.
    window.requestAnimationFrame(scheduleUpdate);

    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("scroll", scheduleUpdate, true);

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", scheduleUpdate);
      window.removeEventListener("scroll", scheduleUpdate, true);
    };
  }, [openInsertMenu]);

  function toggleInsertMenu(nextMenu: OpenInsertMenu): void {
    setOpenInsertMenu((current) => (current?.key === nextMenu.key ? null : nextMenu));
  }

  function closeInsertMenu(): void {
    setOpenInsertMenu(null);
    setInsertMenuPosition(null);
  }

  function openInsertMenuFromEvent(
    event: MouseEvent<HTMLButtonElement>,
    config: Omit<OpenInsertMenu, "anchorEl">
  ): void {
    toggleInsertMenu({
      ...config,
      anchorEl: event.currentTarget,
    });
  }

  function handleToggleKeyDown(
    event: ReactKeyboardEvent<HTMLElement>,
    toggle: () => void
  ): void {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  }

  return {
    openInsertMenu,
    insertMenuPosition,
    insertMenuRef,
    closeInsertMenu,
    openInsertMenuFromEvent,
    handleToggleKeyDown,
  };
}
