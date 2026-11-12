import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The Docker image sets NEXT_OUTPUT=standalone for a self-contained server. It is
  // opt-in because standalone output needs symlink privileges on Windows hosts.
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
