import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "D.A.I.N",
  description: "Distributed Agentic Inference Network",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
