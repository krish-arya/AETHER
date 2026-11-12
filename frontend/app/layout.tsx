import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { Disclaimer } from "@/components/shell/disclaimer";
import { Sidebar } from "@/components/shell/sidebar";
import { Topbar } from "@/components/shell/topbar";

import "./globals.css";
import { Providers } from "./providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: { default: "AETHER", template: "%s · AETHER" },
  description:
    "Project AETHER — quantitative research and decision-support terminal. Not investment advice.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="h-full">
        <Providers>
          <div className="flex h-full flex-col md:flex-row">
            <Sidebar />
            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <Topbar />
              <main className="min-h-0 flex-1 overflow-y-auto p-4 md:p-6">{children}</main>
              <Disclaimer />
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
