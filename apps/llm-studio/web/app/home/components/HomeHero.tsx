import Link from "next/link";
import { FiActivity, FiCpu, FiPlay, FiZap } from "react-icons/fi";

import styles from "../../workspace-home.module.css";

export function HomeHero() {
  return (
    <header className={styles.centeredHeader}>
      <h1 className={styles.heroTitle}>Build better models, faster.</h1>
      <p className={styles.heroSubtitle}>
        The all-in-one workspace for designing LLM architectures, training custom tokenizers,
        and managing your model configurations with ease.
      </p>
      <div className={styles.heroActions}>
        <Link href="/studio" className={styles.primaryButton}>
          <FiZap /> LLM Studio
        </Link>
        <Link href="/tokenizer" className={styles.secondaryButton}>
          <FiCpu /> Tokenizer Studio
        </Link>
        <Link href="/training" className={styles.secondaryButton}>
          <FiActivity /> LLM Training
        </Link>
        <Link href="/inference" className={styles.secondaryButton}>
          <FiPlay /> Inference
        </Link>
      </div>
    </header>
  );
}
