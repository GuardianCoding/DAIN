"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./page.module.css";
import {
  navigation,
  secondaryNavigation,
  type NavItem,
} from "./navigation";

function NavLink({ item }: { item: NavItem }) {
  const pathname = usePathname();
  const Icon = item.icon;

  const active =
    item.href === "/"
      ? pathname === "/"
      : pathname.startsWith(item.href);

  return (
    <Link
      href={item.href}
      className={`${styles.navItem} ${
        active ? styles.active : ""
      }`}
      title={item.description}
    >
      <span className={styles.icon}>
        <Icon size={17} strokeWidth={1.8} />
      </span>

      <span className={styles.label}>{item.label}</span>

      {active && <span className={styles.activeIndicator} />}
    </Link>
  );
}

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <div className={styles.brandMark}>D</div>

        <div className={styles.brandText}>
          <span className={styles.brandName}>D.A.I.N</span>
          <span className={styles.brandSubtitle}>Cluster</span>
        </div>
      </div>

      <nav className={styles.nav}>
        <div className={styles.section}>
          <span className={styles.sectionLabel}>Workspace</span>

          <div className={styles.navList}>
            {navigation.map((item) => (
              <NavLink key={item.href} item={item} />
            ))}
          </div>
        </div>

        <div className={styles.section}>
          <span className={styles.sectionLabel}>System</span>

          <div className={styles.navList}>
            {secondaryNavigation.map((item) => (
              <NavLink key={item.href} item={item} />
            ))}
          </div>
        </div>
      </nav>

      <div className={styles.footer}>
        <div className={styles.connection}>
          <span className={styles.connectionDot} />

          <div>
            <span className={styles.connectionLabel}>Cluster</span>
            <span className={styles.connectionStatus}>Connected</span>
          </div>
        </div>

        <span className={styles.version}>v0.1.0</span>
      </div>
    </aside>
  );
}