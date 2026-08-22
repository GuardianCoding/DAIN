// Home page with splash screen

import styles from "./page.module.css";

export default function Home() {

  return (
    <main className={styles.stage}>
      <p className={styles.kicker}>Distributed Agentic Inference Network</p>
      <h1 className={styles.title}>D.A.I.N</h1>
      <div className={styles.rule} />
      <p className={styles.tag}>Shared pool of computing resources, one agentic AI. Built for the 2026 UQCS Hackathon.</p>
    </main>
  ); 
}
