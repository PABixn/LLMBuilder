"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TrainingMetricPoint } from "../../../lib/trainingApi";
import {
  chartBrushHandlePosition,
  clampMetricRange,
  formatInteger,
  formatMetricAxis,
  formatMetricValue,
  metricAxisDomain,
  metricChartData,
  metricChartStats,
  type MetricChartDatum,
  type MetricChartRange,
} from "../lib/metrics";
import type { MetricChartKey, MetricValueNotation } from "../types";

function MetricChartTooltip({
  active,
  payload,
  label,
  title,
  digits,
  notation = "standard",
}: {
  active?: boolean;
  payload?: Array<{ payload?: { value?: number } }>;
  label?: number | string;
  title: string;
  digits: number;
  notation?: MetricValueNotation;
}) {
  const value = payload?.[0]?.payload?.value;

  if (!active || typeof value !== "number") {
    return null;
  }

  return (
    <div className="trainingChartTooltip">
      <span>Step {label}</span>
      <strong>
        {title}: {formatMetricValue(value, digits, notation)}
      </strong>
    </div>
  );
}

function MetricRangeSelector({
  data,
  range,
  onChange,
}: {
  data: MetricChartDatum[];
  range: MetricChartRange;
  onChange: (range: MetricChartRange) => void;
}) {
  const brushRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    mode: "start" | "end" | "window";
    index: number;
    startIndex: number;
    endIndex: number;
  } | null>(null);
  const [dragMode, setDragMode] = useState<"start" | "end" | "window" | null>(null);
  const lastIndex = data.length - 1;
  const startStep = data[range.startIndex]?.step ?? 0;
  const endStep = data[range.endIndex]?.step ?? 0;
  const selectedLeft = lastIndex > 0 ? (range.startIndex / lastIndex) * 100 : 0;
  const selectedRight = lastIndex > 0 ? (range.endIndex / lastIndex) * 100 : 100;
  const selectedWidth = Math.max(0, selectedRight - selectedLeft);
  const overviewPath = useMemo(() => {
    if (!data.length) {
      return "";
    }
    const width = 1000;
    const height = 72;
    const padding = 7;
    const values = data.map((item) => item.plotValue);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const rangeSize = max - min || 1;
    return data
      .map((item, index) => {
        const x = data.length === 1 ? width / 2 : (index / (data.length - 1)) * width;
        const normalized = (item.plotValue - min) / rangeSize;
        const y = height - padding - normalized * (height - padding * 2);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");
  }, [data]);

  const indexFromEvent = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const bounds = brushRef.current?.getBoundingClientRect();
      if (!bounds || lastIndex <= 0) {
        return 0;
      }
      const ratio = Math.max(0, Math.min((event.clientX - bounds.left) / bounds.width, 1));
      return Math.round(ratio * lastIndex);
    },
    [lastIndex]
  );

  const commitRange = useCallback(
    (startIndex: number, endIndex: number) => {
      const start = Math.max(0, Math.min(startIndex, lastIndex));
      const end = Math.max(start, Math.min(endIndex, lastIndex));
      onChange({ startIndex: start, endIndex: end });
    },
    [lastIndex, onChange]
  );

  const beginDrag = (
    mode: "start" | "end" | "window",
    event: ReactPointerEvent<HTMLButtonElement | HTMLDivElement>
  ) => {
    event.preventDefault();
    event.stopPropagation();
    brushRef.current?.setPointerCapture(event.pointerId);
    const index = indexFromEvent(event as ReactPointerEvent<HTMLDivElement>);
    dragRef.current = {
      mode,
      index,
      startIndex: range.startIndex,
      endIndex: range.endIndex,
    };
    setDragMode(mode);
  };

  const handleTrackPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    const index = indexFromEvent(event);
    const startDistance = Math.abs(index - range.startIndex);
    const endDistance = Math.abs(index - range.endIndex);
    if (index >= range.startIndex && index <= range.endIndex) {
      beginDrag("window", event);
      return;
    }
    commitRange(
      startDistance <= endDistance ? index : range.startIndex,
      startDistance <= endDistance ? range.endIndex : index
    );
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const index = indexFromEvent(event);
    if (drag.mode === "start") {
      commitRange(index, drag.endIndex);
      return;
    }
    if (drag.mode === "end") {
      commitRange(drag.startIndex, index);
      return;
    }

    const windowSize = drag.endIndex - drag.startIndex;
    const delta = index - drag.index;
    const nextStart = Math.max(0, Math.min(drag.startIndex + delta, lastIndex - windowSize));
    commitRange(nextStart, nextStart + windowSize);
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current && event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
    setDragMode(null);
  };

  const handleHandleKeyDown = (
    mode: "start" | "end",
    event: ReactKeyboardEvent<HTMLButtonElement>
  ) => {
    const direction = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
    if (direction === 0) {
      return;
    }
    event.preventDefault();
    const delta = direction * (event.shiftKey ? 10 : 1);
    if (mode === "start") {
      commitRange(range.startIndex + delta, range.endIndex);
      return;
    }
    commitRange(range.startIndex, range.endIndex + delta);
  };

  return (
    <div
      ref={brushRef}
      className={`trainingChartRange ${dragMode ? "isDragging" : ""}`}
      aria-label={`Visible steps ${startStep} to ${endStep}`}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
    >
      <div className="trainingChartRangeMeta">
        <span>Visible steps</span>
        <strong>
          {startStep} - {endStep}
        </strong>
      </div>
      <div className="trainingChartBrush" onPointerDown={handleTrackPointerDown}>
        <svg viewBox="0 0 1000 72" preserveAspectRatio="none" aria-hidden>
          <path d={overviewPath} />
        </svg>
        <div className="trainingChartBrushShade left" style={{ width: `${selectedLeft}%` }} />
        <div
          className="trainingChartBrushSelection"
          style={{ left: `${selectedLeft}%`, width: `${selectedWidth}%` }}
          onPointerDown={(event) => beginDrag("window", event)}
        />
        <button
          type="button"
          className="trainingChartBrushHandle"
          style={{ left: chartBrushHandlePosition(selectedLeft) }}
          aria-label={`Start visible range at step ${startStep}`}
          onPointerDown={(event) => beginDrag("start", event)}
          onKeyDown={(event) => handleHandleKeyDown("start", event)}
        />
        <button
          type="button"
          className="trainingChartBrushHandle"
          style={{ left: chartBrushHandlePosition(selectedRight) }}
          aria-label={`End visible range at step ${endStep}`}
          onPointerDown={(event) => beginDrag("end", event)}
          onKeyDown={(event) => handleHandleKeyDown("end", event)}
        />
        <div className="trainingChartBrushShade right" style={{ left: `${selectedRight}%` }} />
      </div>
    </div>
  );
}

export function MetricChart({
  title,
  metricKey,
  metrics,
  latestValue,
  stroke,
  digits,
}: {
  title: string;
  metricKey: MetricChartKey;
  metrics: TrainingMetricPoint[];
  latestValue: string;
  stroke: string;
  digits: number;
}) {
  const data = useMemo(() => metricChartData(metrics, metricKey), [metricKey, metrics]);
  const previousPointCountRef = useRef(0);
  const [range, setRange] = useState<MetricChartRange | null>(null);

  useEffect(() => {
    setRange((current) => {
      if (!data.length) {
        previousPointCountRef.current = 0;
        return null;
      }

      const previousLastIndex = Math.max(previousPointCountRef.current - 1, 0);
      const nextLastIndex = data.length - 1;
      previousPointCountRef.current = data.length;

      if (!current) {
        return { startIndex: 0, endIndex: nextLastIndex };
      }

      const wasPinnedToEnd = current.endIndex >= previousLastIndex;
      const endIndex = wasPinnedToEnd ? nextLastIndex : Math.min(current.endIndex, nextLastIndex);
      const startIndex = Math.min(current.startIndex, endIndex);
      return { startIndex, endIndex };
    });
  }, [data.length]);

  const visibleRange = clampMetricRange(range, data.length);
  const visibleData = useMemo(
    () => (visibleRange ? data.slice(visibleRange.startIndex, visibleRange.endIndex + 1) : []),
    [data, visibleRange]
  );
  const stats = useMemo(() => metricChartStats(visibleData), [visibleData]);
  const yDomain = useMemo(() => metricAxisDomain(visibleData), [visibleData]);
  const valueNotation: MetricValueNotation = metricKey === "lr" ? "exponential" : "standard";
  const handleRangeChange = useCallback((nextRange: MetricChartRange) => {
    setRange(nextRange);
  }, []);

  return (
    <div className="trainingChartCard">
      <div className="trainingChartHead">
        <div>
          <strong>{title}</strong>
          <span>{data.length ? `${formatInteger(data.length)} points` : "Waiting for data"}</span>
        </div>
        <div className="trainingChartLatest">
          <span>Current</span>
          <strong>{latestValue}</strong>
        </div>
      </div>
      {data.length ? (
        <>
          <div className="trainingChartStats" aria-label={`${title} summary`}>
            <span>Min {formatMetricValue(stats?.min, digits, valueNotation)}</span>
            <span>Max {formatMetricValue(stats?.max, digits, valueNotation)}</span>
            <span>Avg {formatMetricValue(stats?.average, digits, valueNotation)}</span>
          </div>
          <div className="trainingChartFrame">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={visibleData} margin={{ top: 12, right: 12, bottom: 0, left: 4 }}>
                <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="step"
                  tickLine={false}
                  axisLine={false}
                  minTickGap={24}
                  tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                  tickFormatter={(value) => String(value)}
                />
                <YAxis
                  width={58}
                  domain={yDomain}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                  tickFormatter={(value) => formatMetricAxis(Number(value), digits, valueNotation)}
                />
                <Tooltip
                  cursor={{ stroke: "var(--text-muted)", strokeDasharray: "4 4" }}
                  content={
                    <MetricChartTooltip
                      title={title}
                      digits={digits}
                      notation={valueNotation}
                    />
                  }
                />
                <Line
                  type="monotone"
                  dataKey="plotValue"
                  stroke={stroke}
                  strokeWidth={2.4}
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 2, stroke }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {data.length > 8 && visibleRange ? (
            <MetricRangeSelector data={data} range={visibleRange} onChange={handleRangeChange} />
          ) : null}
        </>
      ) : (
        <div className="trainingEmpty">Metrics will appear after the first logged steps.</div>
      )}
    </div>
  );
}
