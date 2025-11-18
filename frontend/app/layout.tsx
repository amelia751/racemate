import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { LiveKitProvider } from "@/lib/livekit/LiveKitContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RaceMate - Real-Time Race Strategy Platform",
  description: "AI-powered voice agent for real-time racing strategy, fuel management, and tire analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-screen">
      <body className={`${inter.className} bg-background h-screen overflow-hidden`}>
        <LiveKitProvider>
          {children}
        </LiveKitProvider>
      </body>
    </html>
  );
}
