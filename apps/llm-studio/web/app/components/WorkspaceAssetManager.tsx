"use client";

import { useState } from "react";
import { FiArchive, FiDownload, FiFolder, FiSearch } from "react-icons/fi";

import {
  formatAge,
  formatBytes,
  formatDate,
  type WorkspaceAssetInventory,
} from "../../lib/workspaceAssets";
import styles from "../workspace-home.module.css";

interface WorkspaceAssetManagerProps {
  inventory: WorkspaceAssetInventory;
  title: string;
  description?: string;
}

export function WorkspaceAssetManager({
  inventory,
  title,
  description,
}: WorkspaceAssetManagerProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const normalizedQuery = searchQuery.trim().toLowerCase();
  const showLoadingState = inventory.loading && inventory.lastRefreshedAt === null;
  const filteredAssets =
    normalizedQuery === ""
      ? inventory.assets
      : inventory.assets.filter((asset) => {
          return (
            asset.name.toLowerCase().includes(normalizedQuery) ||
            asset.type.toLowerCase().includes(normalizedQuery)
          );
        });

  return (
    <section>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionLead}>
          <h2 className={styles.sectionTitle}>{title}</h2>
          {description ? <p className={styles.sectionCopy}>{description}</p> : null}
        </div>
        <div className={styles.searchWrapper}>
          <FiSearch className={styles.searchIcon} />
          <input
            type="text"
            placeholder="Search assets..."
            className={styles.searchInput}
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
          />
        </div>
      </div>

      <div className={styles.assetGrid}>
        {showLoadingState
          ? Array.from({ length: 4 }, (_, index) => (
              <div key={`loading-${index}`} className={styles.loadingRow} aria-hidden="true">
                <div className={styles.assetHeader}>
                  <div className={`${styles.assetIcon} ${styles.loadingBlock}`} />
                  <div className={styles.loadingText}>
                    <span className={`${styles.loadingLine} ${styles.loadingLineWide}`} />
                    <span className={`${styles.loadingLine} ${styles.loadingLineMedium}`} />
                  </div>
                </div>
                <div className={styles.assetFooter}>
                  <span className={`${styles.loadingLine} ${styles.loadingLineNarrow}`} />
                  <span className={`${styles.assetTag} ${styles.loadingBlock} ${styles.loadingTag}`} />
                </div>
              </div>
            ))
          : filteredAssets.map((asset) => (
              <div key={`${asset.type}-${asset.id}`} className={styles.assetCard}>
                <div className={styles.assetHeader}>
                  <div className={styles.assetIcon}>
                    {asset.type === "model" ? <FiFolder /> : <FiArchive />}
                  </div>
                  <div className={styles.assetMain}>
                    <span className={styles.assetName}>{asset.name}</span>
                    <span className={styles.assetMeta}>
                      {asset.type === "model" ? "Model Config" : "Tokenizer Artifact"} •{" "}
                      {formatAge(asset.createdAt)}
                    </span>
                  </div>
                </div>
                <div className={styles.assetFooter}>
                  <span className={styles.assetMeta} style={{ fontSize: "0.7rem" }}>
                    {formatDate(asset.createdAt)}
                  </span>
                  <span className={styles.assetTag}>
                    {asset.type === "model" ? formatBytes(asset.size ?? 0) : asset.status}
                  </span>
                </div>
                {asset.downloadUrl ? (
                  <a
                    href={asset.downloadUrl}
                    download={asset.fileName ?? undefined}
                    className={styles.downloadButton}
                    title="Download"
                  >
                    <FiDownload />
                  </a>
                ) : null}
              </div>
            ))}

        {!showLoadingState && filteredAssets.length === 0 ? (
          <div className={styles.emptyState}>
            <p className={styles.heroSubtitle}>
              {searchQuery ? "No assets match your search." : "No assets found in this workspace."}
            </p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
