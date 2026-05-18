"use client";

import {
  Children,
  isValidElement,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type FocusEvent,
  type MouseEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { FiInfo } from "react-icons/fi";

type HelpTooltipProps = {
  label: string;
  children: ReactNode;
  content: ReactNode;
  align?: "left" | "right" | "center";
  width?: "default" | "wide";
  title?: ReactNode;
};

type TooltipAlign = NonNullable<HelpTooltipProps["align"]>;
type TooltipWidth = NonNullable<HelpTooltipProps["width"]>;

type TooltipPosition = {
  style: CSSProperties;
  placement: "above" | "below";
};

const TOOLTIP_VIEWPORT_PADDING = 16;
const TOOLTIP_GAP = 9;
const TOOLTIP_ABOVE_MIN_SPACE = 170;
const TOOLTIP_WIDTHS: Record<TooltipWidth, number> = {
  default: 280,
  wide: 380,
};

export function HelpTooltip({
  label,
  children,
  content,
  align = "center",
  width = "default",
  title,
}: HelpTooltipProps) {
  const tooltipId = useId();
  const tooltipTitle = title ?? inferTooltipTitle(label);
  const hasProvidedTitle = tooltipContentHasTitle(content);
  const body = (
    <>
      {hasProvidedTitle ? null : <strong>{tooltipTitle}</strong>}
      {renderTooltipBody(content)}
    </>
  );

  return (
    <TooltipShell
      tooltipId={tooltipId}
      label={label}
      content={body}
      align={align}
      width={width}
      describedByOnWrapper
    >
      {children}
    </TooltipShell>
  );
}

type InfoTooltipProps = {
  label: string;
  children: ReactNode;
  align?: "left" | "right" | "center";
  width?: "default" | "wide";
  title?: ReactNode;
};

export function InfoTooltip({
  label,
  children,
  align = "center",
  width = "default",
  title,
}: InfoTooltipProps) {
  const tooltipId = useId();
  const tooltipTitle = title ?? inferTooltipTitle(label);
  const hasProvidedTitle = tooltipContentHasTitle(children);
  const body = (
    <>
      {hasProvidedTitle ? null : <strong>{tooltipTitle}</strong>}
      {renderTooltipBody(children)}
    </>
  );

  return (
    <TooltipShell tooltipId={tooltipId} label={label} content={body} align={align} width={width}>
      <button
        type="button"
        className="helpTooltipIcon"
        aria-label={label}
        aria-describedby={tooltipId}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
        }}
      >
        <FiInfo aria-hidden="true" />
      </button>
    </TooltipShell>
  );
}

type TooltipShellProps = {
  tooltipId: string;
  label: string;
  children: ReactNode;
  content: ReactNode;
  align: TooltipAlign;
  width: TooltipWidth;
  describedByOnWrapper?: boolean;
};

function TooltipShell({
  tooltipId,
  label,
  children,
  content,
  align,
  width,
  describedByOnWrapper = false,
}: TooltipShellProps) {
  const triggerRef = useRef<HTMLSpanElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<TooltipPosition | null>(null);

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;

    if (!trigger || typeof window === "undefined") {
      return;
    }

    setPosition(getTooltipPosition(trigger.getBoundingClientRect(), align, width));
  }, [align, width]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);

    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [isOpen, updatePosition]);

  const openTooltip = useCallback(() => {
    setIsOpen(true);
    updatePosition();
  }, [updatePosition]);

  const closeTooltip = useCallback(() => {
    setIsOpen(false);
  }, []);

  const handleBlur = useCallback(
    (event: FocusEvent<HTMLSpanElement>) => {
      if (event.currentTarget.contains(event.relatedTarget)) {
        return;
      }

      closeTooltip();
    },
    [closeTooltip],
  );

  const handleMouseLeave = useCallback(
    (_event: MouseEvent<HTMLSpanElement>) => {
      closeTooltip();
    },
    [closeTooltip],
  );

  const bubbleClassName = useMemo(() => {
    return [
      "helpTooltipBubble",
      "helpTooltipBubblePortal",
      `helpTooltipBubble-${position?.placement ?? "above"}`,
    ].join(" ");
  }, [position?.placement]);

  return (
    <span
      ref={triggerRef}
      className={`helpTooltip helpTooltip-${align} helpTooltip-${width}`}
      aria-describedby={describedByOnWrapper ? tooltipId : undefined}
      data-tooltip-label={label}
      onMouseEnter={openTooltip}
      onMouseLeave={handleMouseLeave}
      onFocus={openTooltip}
      onBlur={handleBlur}
    >
      {children}
      {isOpen && position
        ? createPortal(
            <span
              id={tooltipId}
              className={bubbleClassName}
              role="tooltip"
              style={position.style}
            >
              {content}
            </span>,
            document.body,
          )
        : null}
    </span>
  );
}

function getTooltipPosition(
  triggerRect: DOMRect,
  align: TooltipAlign,
  width: TooltipWidth,
): TooltipPosition {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const tooltipWidth = Math.min(
    TOOLTIP_WIDTHS[width],
    Math.max(180, viewportWidth - TOOLTIP_VIEWPORT_PADDING * 2),
  );
  const maxLeft = Math.max(TOOLTIP_VIEWPORT_PADDING, viewportWidth - tooltipWidth - TOOLTIP_VIEWPORT_PADDING);
  const placement = triggerRect.top < TOOLTIP_ABOVE_MIN_SPACE ? "below" : "above";
  let left = triggerRect.left + triggerRect.width / 2 - tooltipWidth / 2;

  if (align === "left") {
    left = triggerRect.left;
  } else if (align === "right") {
    left = triggerRect.right - tooltipWidth;
  }

  left = Math.min(Math.max(TOOLTIP_VIEWPORT_PADDING, left), maxLeft);

  const verticalStyle =
    placement === "below"
      ? { top: Math.min(viewportHeight - TOOLTIP_VIEWPORT_PADDING, triggerRect.bottom + TOOLTIP_GAP) }
      : { bottom: Math.max(TOOLTIP_VIEWPORT_PADDING, viewportHeight - triggerRect.top + TOOLTIP_GAP) };

  return {
    placement,
    style: {
      ...verticalStyle,
      left,
      width: tooltipWidth,
    },
  };
}

function inferTooltipTitle(label: string) {
  return label
    .replace(/\s+explanation$/i, "")
    .replace(/\s+tooltip$/i, "")
    .trim();
}

function tooltipContentHasTitle(content: ReactNode) {
  const firstContentNode = Children.toArray(content).find((child) => {
    return typeof child !== "string" || child.trim() !== "";
  });

  return isValidElement(firstContentNode) && firstContentNode.type === "strong";
}

function renderTooltipBody(content: ReactNode) {
  if (typeof content === "string") {
    return <p>{content}</p>;
  }

  return content;
}

type FieldLabelTextProps = {
  children: ReactNode;
  tooltip: ReactNode;
  tooltipLabel: string;
  align?: "left" | "right" | "center";
  width?: "default" | "wide";
};

export function FieldLabelText({
  children,
  tooltip,
  tooltipLabel,
  align = "left",
  width = "default",
}: FieldLabelTextProps) {
  return (
    <span className="fieldLabelText">
      <span>{children}</span>
      <InfoTooltip label={tooltipLabel} align={align} width={width}>
        {tooltip}
      </InfoTooltip>
    </span>
  );
}
