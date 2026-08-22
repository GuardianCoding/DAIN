import Sidebar from "../Sidebar/sidebar";
import styles from "./page.module.css";

export default function AppShell({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className={styles.shell}>
      <Sidebar />

      <main className={styles.content}>
        {children}
      </main>
    </div>
  );
}