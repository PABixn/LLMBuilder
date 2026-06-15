"use client";

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type DragEvent,
  type FocusEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

const STATUS_TOOLTIP_VIEWPORT_MARGIN = 12;
const STATUS_TOOLTIP_GAP = 10;
const STATUS_TOOLTIP_MAX_WIDTH = 360;
const STATUS_TOOLTIP_MIN_HEIGHT = 140;

type StatusTooltipPlacement = "top" | "bottom";

interface StatusTooltipPosition {
  top: number;
  left: number;
  width: number;
  maxHeight: number;
  arrowLeft: number;
  placement: StatusTooltipPlacement;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function DropSlot({
  active,
  compact,
  label,
  onDragOver,
  onDrop,
}: {
  active: boolean;
  compact?: boolean;
  label: string;
  onDragOver: (event: DragEvent<HTMLDivElement>) => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
}) {
  return (
    <div
      className={`dropSlot${compact ? " isCompact" : ""}${active ? " isActive" : ""}`}
      onDragOver={onDragOver}
      onDrop={onDrop}
      aria-label={label}
      title={label}
    >
      <span className="dropSlotMark" aria-hidden />
    </div>
  );
}

export function PaletteTile({
  title,
  subtitle,
  colorClass,
  draggable,
  onDragStart,
  onDragEnd,
  hint,
}: {
  title: string;
  subtitle: string;
  colorClass: string;
  draggable: boolean;
  onDragStart?: (event: DragEvent<HTMLDivElement>) => void;
  onDragEnd?: () => void;
  hint?: string;
}) {
  return (
    <div
      className={`paletteTile ${colorClass}`}
      draggable={draggable}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
    >
      <div className="paletteTileTitle">{title}</div>
      <div className="paletteTileSubtitle">{subtitle}</div>
      <div className="paletteTileHint">{hint ?? (draggable ? "Drag to canvas" : "Add from slot menu")}</div>
    </div>
  );
}

export function StatusCard({
  title,
  value,
  detail,
  tone,
  icon,
  tooltipContent,
  tooltipLabel: _tooltipLabel,
}: {
  title: string;
  value: string;
  detail: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  icon: ReactNode;
  tooltipContent?: ReactNode;
  tooltipLabel?: string;
}) {
  const tooltipId = useId();
  const hasTooltip = tooltipContent != null;
  const cardRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<StatusTooltipPosition | null>(null);

  const clearCloseTimer = useCallback(() => {
    if (closeTimerRef.current !== null) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const closeTooltip = useCallback(() => {
    clearCloseTimer();
    setTooltipOpen(false);
  }, [clearCloseTimer]);

  const scheduleCloseTooltip = useCallback(() => {
    clearCloseTimer();
    closeTimerRef.current = setTimeout(() => {
      setTooltipOpen(false);
    }, 110);
  }, [clearCloseTimer]);

  const openTooltip = useCallback(() => {
    if (!hasTooltip) {
      return;
    }
    clearCloseTimer();
    if (!tooltipOpen) {
      setTooltipPosition(null);
    }
    setTooltipOpen(true);
  }, [clearCloseTimer, hasTooltip, tooltipOpen]);

  const updateTooltipPosition = useCallback(() => {
    const card = cardRef.current;
    if (!card || typeof window === "undefined") {
      return;
    }

    const triggerRect = card.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const width = Math.min(
      STATUS_TOOLTIP_MAX_WIDTH,
      viewportWidth - STATUS_TOOLTIP_VIEWPORT_MARGIN * 2
    );
    const tooltipHeight = tooltipRef.current?.offsetHeight ?? 240;
    const topSpace =
      triggerRect.top - STATUS_TOOLTIP_VIEWPORT_MARGIN - STATUS_TOOLTIP_GAP;
    const bottomSpace =
      viewportHeight - triggerRect.bottom - STATUS_TOOLTIP_VIEWPORT_MARGIN - STATUS_TOOLTIP_GAP;
    const placement: StatusTooltipPlacement =
      bottomSpace >= Math.min(tooltipHeight, STATUS_TOOLTIP_MIN_HEIGHT) ||
      bottomSpace >= topSpace
        ? "bottom"
        : "top";
    const availableHeight = Math.max(
      STATUS_TOOLTIP_MIN_HEIGHT,
      placement === "bottom" ? bottomSpace : topSpace
    );
    const renderedHeight = Math.min(tooltipHeight, availableHeight);
    const left = clamp(
      triggerRect.left + triggerRect.width / 2 - width / 2,
      STATUS_TOOLTIP_VIEWPORT_MARGIN,
      viewportWidth - width - STATUS_TOOLTIP_VIEWPORT_MARGIN
    );
    const top =
      placement === "bottom"
        ? Math.min(
            viewportHeight - STATUS_TOOLTIP_VIEWPORT_MARGIN - renderedHeight,
            triggerRect.bottom + STATUS_TOOLTIP_GAP
          )
        : Math.max(
            STATUS_TOOLTIP_VIEWPORT_MARGIN,
            triggerRect.top - STATUS_TOOLTIP_GAP - renderedHeight
          );
    const arrowLeft = clamp(
      triggerRect.left + triggerRect.width / 2 - left,
      16,
      width - 16
    );

    setTooltipPosition({
      top,
      left,
      width,
      maxHeight: availableHeight,
      arrowLeft,
      placement,
    });
  }, []);

  useEffect(() => {
    if (!tooltipOpen) {
      return undefined;
    }

    updateTooltipPosition();
    const frame = window.requestAnimationFrame(updateTooltipPosition);
    window.addEventListener("resize", updateTooltipPosition);
    window.addEventListener("scroll", updateTooltipPosition, true);

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        closeTooltip();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", updateTooltipPosition);
      window.removeEventListener("scroll", updateTooltipPosition, true);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [closeTooltip, tooltipOpen, updateTooltipPosition]);

  useEffect(() => {
    return () => clearCloseTimer();
  }, [clearCloseTimer]);

  function handleBlur(event: FocusEvent<HTMLDivElement>): void {
    const nextTarget = event.relatedTarget;
    if (
      nextTarget instanceof Node &&
      (cardRef.current?.contains(nextTarget) || tooltipRef.current?.contains(nextTarget))
    ) {
      return;
    }
    closeTooltip();
  }

  const tooltipStyle = tooltipPosition
    ? ({
        top: tooltipPosition.top,
        left: tooltipPosition.left,
        width: tooltipPosition.width,
        maxHeight: tooltipPosition.maxHeight,
        "--status-card-tooltip-arrow-left": `${tooltipPosition.arrowLeft}px`,
      } as CSSProperties)
    : undefined;

  return (
    <div
      ref={cardRef}
      className={`statusCard${tone ? ` tone-${tone}` : ""}${hasTooltip ? " hasTooltip" : ""}`}
      tabIndex={hasTooltip ? 0 : undefined}
      aria-describedby={hasTooltip ? tooltipId : undefined}
      onMouseEnter={openTooltip}
      onMouseLeave={hasTooltip ? scheduleCloseTooltip : undefined}
      onFocus={openTooltip}
      onBlur={hasTooltip ? handleBlur : undefined}
    >
      <div className="statusCardIcon" aria-hidden>
        {icon}
      </div>
      <div>
        <div className="statusCardTitle">{title}</div>
        <div className="statusCardValue">{value}</div>
        <div className="statusCardDetail">{detail}</div>
      </div>
      {hasTooltip && tooltipOpen
        ? createPortal(
            <div
              ref={tooltipRef}
              id={tooltipId}
              className={`statusCardTooltip is-${tooltipPosition?.placement ?? "bottom"}${
                tooltipPosition ? "" : " is-hidden"
              }`}
              role="tooltip"
              style={tooltipStyle}
              onMouseEnter={openTooltip}
              onMouseLeave={scheduleCloseTooltip}
              onFocus={openTooltip}
              onBlur={handleBlur}
            >
              {tooltipContent}
            </div>,
            document.body
          )
        : null}
    </div>
  );
}
