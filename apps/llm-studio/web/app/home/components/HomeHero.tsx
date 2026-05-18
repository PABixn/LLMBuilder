import Link from "next/link";
import { FiActivity, FiCpu, FiPlay, FiZap } from "react-icons/fi";

import styles from "../../workspace-home.module.css";

export function HomeHero() {
  return (
    <header className={styles.centeredHeader}>
      <h1 className={styles.heroTitle}>Build models faster.</h1>
      <p className={styles.heroSubtitle}>
        Design models, train tokenizers, run training, and test outputs in one place.
      </p>
      <div className={styles.heroActions}>
        <Link href="/studio" className={styles.primaryButton}>
          <FiZap /> Model Studio
        </Link>
        <Link href="/tokenizer" className={styles.secondaryButton}>
          <FiCpu /> Tokenizer
        </Link>
        <Link href="/training" className={styles.secondaryButton}>
          <FiActivity /> Training
        </Link>
        <Link href="/inference" className={styles.secondaryButton}>
          <FiPlay /> Inference
        </Link>
      </div>
    </header>
  );
}
