import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Proxy API calls to Python backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/gpu/:path*",
        destination: "http://localhost:8000/gpu/:path*",
      },
    ];
  },
};

export default nextConfig;
