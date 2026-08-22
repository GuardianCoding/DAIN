// lib/feed/useJobs.ts
// Thin selector over FeedProvider's accumulated job state. All parsing,
// accumulation, and stale-job refresh happens once, in the provider — this
// just reads it and re-exposes seedJob for the Create Job page.

"use client";

import { useFeed } from "./FeedProvider";

export function useJobs() {
  const { jobs, seedJob } = useFeed();
  return { jobs, seedJob };
}