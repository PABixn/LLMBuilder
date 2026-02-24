import type { DragEvent, ReactNode } from "react";

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
}: {
  title: string;
  value: string;
  detail: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  icon: ReactNode;
}) {
  return (
    <div className={`statusCard${tone ? ` tone-${tone}` : ""}`}>
      <div className="statusCardIcon" aria-hidden>
        {icon}
      </div>
      <div>
        <div className="statusCardTitle">{title}</div>
        <div className="statusCardValue">{value}</div>
        <div className="statusCardDetail">{detail}</div>
      </div>
    </div>
  );
}
