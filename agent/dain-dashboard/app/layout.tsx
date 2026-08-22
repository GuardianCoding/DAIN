import "./globals.css";
import AppShell from "./components/AppShell/AppShell";
import { FeedProvider } from "../lib/feed/FeedProvider";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <FeedProvider>
          <AppShell>{children}</AppShell>
        </FeedProvider>
      </body>
    </html>
  );
}