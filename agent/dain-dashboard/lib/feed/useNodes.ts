// lib/feed/useNodes.ts
// Thin selector over FeedProvider's accumulated node state. All parsing and
// accumulation happens once, in the provider — this just reads it.

"use client";

import { useFeed } from "./FeedProvider";

export function useNodes() {
  return useFeed().nodes;
}