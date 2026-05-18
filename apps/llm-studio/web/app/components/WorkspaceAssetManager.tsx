"use client";

import { useState, useMemo } from "react";
import { 
  FiDownload, 
  FiSearch, 
  FiFilter, 
  FiTrash2, 
  FiChevronDown,
  FiBox,
  FiCpu,
  FiActivity,
  FiLayers,
  FiArrowRight
} from "react-icons/fi";
import Link from "next/link";
import { useRouter } from "next/navigation";

import {
  formatAge,
  formatBytes,
  type WorkspaceAsset,
  type WorkspaceAssetInventory,
} from "../../lib/workspaceAssets";
import { HelpTooltip, InfoTooltip } from "../shared/components/HelpTooltip";
import styles from "../workspace-home.module.css";

interface WorkspaceAssetManagerProps {
  inventory: WorkspaceAssetInventory;
  title: string;
  description?: string;
  selectedModelId?: string | null;
  selectedTokenizerId?: string | null;
  onUseAsModel?: (asset: WorkspaceAsset) => void;
  onUseAsTokenizer?: (asset: WorkspaceAsset) => void;
}

type FilterType = "all" | "model" | "tokenizer" | "training_run";
type SortBy = "date-desc" | "date-asc" | "name-asc" | "name-desc" | "size-desc";

export function WorkspaceAssetManager({
  inventory,
  title,
  description,
  selectedModelId = null,
  selectedTokenizerId = null,
  onUseAsModel,
  onUseAsTokenizer,
}: WorkspaceAssetManagerProps) {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<FilterType>("all");
  const [sortBy, setSortBy] = useState<SortBy>("date-desc");

  const normalizedQuery = searchQuery.trim().toLowerCase();
  const hasAssets = inventory.assets.length > 0;
  const showLoadingState = inventory.loading && inventory.lastRefreshedAt === null;

  const filteredAndSortedAssets = useMemo(() => {
    let result = [...inventory.assets];

    if (normalizedQuery !== "") {
      result = result.filter((asset) => {
        return (
          asset.name.toLowerCase().includes(normalizedQuery) ||
          asset.type.toLowerCase().includes(normalizedQuery) ||
          (asset.status && asset.status.toLowerCase().includes(normalizedQuery))
        );
      });
    }

    if (filterType !== "all") {
      result = result.filter((asset) => asset.type === filterType);
    }

    result.sort((a, b) => {
      switch (sortBy) {
        case "date-desc":
          return Date.parse(b.createdAt) - Date.parse(a.createdAt);
        case "date-asc":
          return Date.parse(a.createdAt) - Date.parse(b.createdAt);
        case "name-asc":
          return a.name.localeCompare(b.name);
        case "name-desc":
          return b.name.localeCompare(a.name);
        case "size-desc":
          return (b.size ?? 0) - (a.size ?? 0);
        default:
          return 0;
      }
    });

    return result;
  }, [inventory.assets, normalizedQuery, filterType, sortBy]);

  const handleDelete = async (asset: WorkspaceAsset, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Delete "${asset.name}"? This cannot be undone.`)) {
      try {
        await inventory.deleteAsset(asset);
      } catch (err) {
        alert(err instanceof Error ? err.message : "Could not delete this item.");
      }
    }
  };

  const handleRemoveAll = async () => {
    if (confirm("Delete all workspace items? This cannot be undone.")) {
      try {
        await inventory.deleteAllAssets();
      } catch (err) {
        alert(err instanceof Error ? err.message : "Could not delete all items.");
      }
    }
  };

  const handleCardClick = (asset: WorkspaceAsset) => {
    const url =
      asset.type === "model"
        ? `/studio?project=${asset.id}`
        : asset.type === "tokenizer"
          ? `/tokenizer?job=${asset.id}`
          : `/training?run=${asset.id}`;
    router.push(url);
  };

  const getStatusClass = (status?: string) => {
    if (!status) return "";
    const s = status.toUpperCase();
    if (s === "COMPLETED" || s === "READY") return styles.tagCompleted;
    if (s === "FAILED") return styles.tagFailed;
    if (s === "RUNNING" || s === "TRAINING") return styles.tagRunning;
    if (s === "PENDING" || s === "QUEUED") return styles.tagPending;
    return "";
  };

  return (
    <section className={styles.assetManagerSection}>
      <div className={styles.sectionHeader}>
        <div className={styles.controlsRow} style={{ marginBottom: "12px", justifyContent: "space-between" }}>
          <div className={styles.sectionLead}>
            <h2 className={styles.sectionTitle}>
              {title}
              <InfoTooltip label="Workspace explanation" align="left" width="wide">
                <strong>Workspace</strong>
                <p>
                  This list combines saved model configs, tokenizer jobs, and training runs.
                  Selecting a card opens the page that owns that asset.
                </p>
              </InfoTooltip>
            </h2>
            {description ? <p className={styles.sectionCopy}>{description}</p> : null}
          </div>
          {hasAssets && (
            <HelpTooltip label="Delete all workspace items" content="Deletes every workspace item shown by the backend inventory. This cannot be undone.">
              <button
                className={styles.removeAllButton}
                onClick={handleRemoveAll}
                disabled={inventory.refreshing}
              >
                <FiTrash2 /> Delete all
              </button>
            </HelpTooltip>
          )}
        </div>

        {hasAssets && (
          <div className={styles.controlsRow}>
            <div className={styles.searchWrapper}>
              <FiSearch className={styles.searchIcon} />
              <input
                type="text"
                placeholder="Search workspace"
                className={styles.searchInput}
                value={searchQuery}
                aria-label="Search workspace assets"
                onChange={(event) => setSearchQuery(event.target.value)}
              />
            </div>
            <div className={styles.filterControls}>
              <div className={styles.selectWrapper}>
                <FiLayers className={styles.controlIcon} />
                <HelpTooltip label="Workspace type filter" content="Filters the inventory by asset type: model configs, tokenizer artifacts, or training runs.">
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value as FilterType)}
                    className={styles.controlSelect}
                    aria-label="Filter workspace assets"
                  >
                    <option value="all">All items</option>
                    <option value="model">Models</option>
                    <option value="tokenizer">Tokenizers</option>
                    <option value="training_run">Training runs</option>
                  </select>
                </HelpTooltip>
                <FiChevronDown className={styles.chevronIcon} />
              </div>
              <div className={styles.selectWrapper}>
                <FiFilter className={styles.controlIcon} />
                <HelpTooltip label="Workspace sort order" content="Changes how matching assets are ordered. Largest uses known artifact size when available.">
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as SortBy)}
                    className={styles.controlSelect}
                    aria-label="Sort workspace assets"
                  >
                    <option value="date-desc">Newest</option>
                    <option value="date-asc">Oldest</option>
                    <option value="name-asc">Name A-Z</option>
                    <option value="name-desc">Name Z-A</option>
                    <option value="size-desc">Largest</option>
                  </select>
                </HelpTooltip>
                <FiChevronDown className={styles.chevronIcon} />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className={styles.assetGrid}>
        {showLoadingState
          ? Array.from({ length: 3 }, (_, index) => (
              <div key={`loading-${index}`} className={styles.loadingRow} aria-hidden="true">
                <div className={styles.assetHeader}>
                  <div className={`${styles.assetIcon} ${styles.loadingBlock}`} />
                  <div className={styles.loadingText}>
                    <span className={`${styles.loadingLine} ${styles.loadingLineWide}`} />
                    <span className={`${styles.loadingLine} ${styles.loadingLineMedium}`} />
                  </div>
                </div>
                <div className={styles.assetInfo}>
                  <span className={`${styles.loadingLine} ${styles.loadingLineNarrow}`} />
                </div>
              </div>
            ))
          : hasAssets ? (
              filteredAndSortedAssets.length > 0 ? (
                filteredAndSortedAssets.map((asset) => (
                  <div 
                    key={`${asset.type}-${asset.id}`} 
                    className={styles.assetCard}
                    onClick={() => handleCardClick(asset)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => e.key === 'Enter' && handleCardClick(asset)}
                  >
                    <div className={styles.assetHeader}>
                      <div className={styles.assetIcon}>
                        {asset.type === "model" ? <FiBox /> : asset.type === "tokenizer" ? <FiCpu /> : <FiActivity />}
                      </div>
                      <div className={styles.assetMain}>
                        <span className={styles.assetName}>{asset.name}</span>
                        <div className={styles.assetMeta}>
                          <span>
                            {asset.subtitle ??
                              (asset.type === "model"
                                ? "Model config"
                                : asset.type === "tokenizer"
                                  ? "Tokenizer"
                                  : "Training run")}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className={styles.assetInfo}>
                       <div className={styles.assetAge}>
                        {formatAge(asset.createdAt)}
                      </div>
                      <span className={`${styles.assetTag} ${asset.type !== "model" || asset.status === "READY" ? getStatusClass(asset.status) : ""}`}>
                        {asset.type === "model"
                          ? formatBytes(asset.size ?? 0)
                          : asset.status ?? formatBytes(asset.size ?? 0)}
                      </span>
                    </div>

                    <div className={styles.assetActions} onClick={(e) => e.stopPropagation()}>
                      {asset.type === "model" && onUseAsModel ? (
                        <HelpTooltip label="Use model asset" content="Selects this saved model config for the current workflow without opening the asset card.">
                          <button
                            type="button"
                            className={`${styles.assetSelectButton} ${selectedModelId === asset.id ? styles.assetSelectButtonActive : ""}`}
                            onClick={() => onUseAsModel(asset)}
                          >
                            {selectedModelId === asset.id ? "Selected" : "Use model"}
                          </button>
                        </HelpTooltip>
                      ) : null}

                      {asset.type === "tokenizer" && onUseAsTokenizer ? (
                        <HelpTooltip label="Use tokenizer asset" content="Selects this completed tokenizer for the current workflow. Incomplete tokenizer jobs cannot be used for model training.">
                          <button
                            type="button"
                            className={`${styles.assetSelectButton} ${selectedTokenizerId === asset.id ? styles.assetSelectButtonActive : ""}`}
                            onClick={() => onUseAsTokenizer(asset)}
                            disabled={asset.status !== "COMPLETED"}
                          >
                            {selectedTokenizerId === asset.id ? "Selected" : "Use tokenizer"}
                          </button>
                        </HelpTooltip>
                      ) : null}

                      {asset.downloadUrl ? (
                        <HelpTooltip label="Download asset" content="Downloads this asset bundle or file from the backend when a download URL is available.">
                          <a
                            href={asset.downloadUrl}
                            download={asset.fileName ?? undefined}
                            className={styles.actionButton}
                            aria-label={`Download ${asset.name}`}
                          >
                            <FiDownload />
                          </a>
                        </HelpTooltip>
                      ) : null}

                      <HelpTooltip label="Delete asset" content="Deletes this workspace item after confirmation. The action cannot be undone.">
                        <button
                          onClick={(e) => handleDelete(asset, e)}
                          className={`${styles.actionButton} ${styles.actionButtonDanger}`}
                          aria-label={`Delete ${asset.name}`}
                        >
                          <FiTrash2 />
                        </button>
                      </HelpTooltip>
                    </div>
                  </div>
                ))
              ) : (
                <div className={styles.emptyState}>
                  <FiSearch style={{ fontSize: "3.5rem", color: "var(--text-muted)", opacity: 0.2, marginBottom: "8px" }} />
                  <h3 className={styles.emptyStateTitle}>No matches</h3>
                  <p className={styles.heroSubtitle}>
                    No items match "{searchQuery}".
                  </p>
                </div>
              )
            ) : (
              <div className={styles.emptyState}>
                <h3 className={styles.emptyStateTitle}>Your workspace is empty</h3>
                <p className={styles.heroSubtitle}>
                  Create a model, tokenizer, or training run to get started.
                </p>
                <div className={styles.emptyStateGrid}>
                  <Link href="/studio" className={styles.emptyActionCard}>
                    <div className={styles.emptyActionIcon}>
                      <FiBox />
                    </div>
                    <div className={styles.emptyActionContent}>
                      <h4 className={styles.emptyActionTitle}>Design model</h4>
                      <p className={styles.emptyActionText}>
                        Set layers and model settings.
                      </p>
                    </div>
                    <FiArrowRight style={{ marginTop: "auto", fontSize: "1.2rem", opacity: 0.4 }} />
                  </Link>
                  <Link href="/tokenizer" className={styles.emptyActionCard}>
                    <div className={styles.emptyActionIcon}>
                      <FiCpu />
                    </div>
                    <div className={styles.emptyActionContent}>
                      <h4 className={styles.emptyActionTitle}>Train tokenizer</h4>
                      <p className={styles.emptyActionText}>
                        Build a vocabulary from your data.
                      </p>
                    </div>
                    <FiArrowRight style={{ marginTop: "auto", fontSize: "1.2rem", opacity: 0.4 }} />
                  </Link>
                  <Link href="/training" className={styles.emptyActionCard}>
                    <div className={styles.emptyActionIcon}>
                      <FiActivity />
                    </div>
                    <div className={styles.emptyActionContent}>
                      <h4 className={styles.emptyActionTitle}>Start training</h4>
                      <p className={styles.emptyActionText}>
                        Run training and track progress.
                      </p>
                    </div>
                    <FiArrowRight style={{ marginTop: "auto", fontSize: "1.2rem", opacity: 0.4 }} />
                  </Link>
                </div>
              </div>
            )}
      </div>
    </section>
  );
}
