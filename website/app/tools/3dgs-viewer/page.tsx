"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";

const SplatViewer = dynamic(() => import("@/app/components/SplatViewer"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm bg-black">
      Loading viewer…
    </div>
  ),
});

const ACCEPTED_EXTENSIONS = [".ply", ".spz", ".splat", ".ksplat"];

export default function SplatViewerPage() {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [blobUrl]);

  const acceptFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const lowerName = file.name.toLowerCase();
      const ok = ACCEPTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
      if (!ok) {
        setError(
          `Unsupported file type. Expected one of: ${ACCEPTED_EXTENSIONS.join(", ")}`
        );
        return;
      }
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      setError(null);
      setFileName(file.name);
      setBlobUrl(URL.createObjectURL(file));
    },
    [blobUrl]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      acceptFile(e.dataTransfer.files?.[0]);
    },
    [acceptFile]
  );

  const reset = useCallback(() => {
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    setBlobUrl(null);
    setFileName(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }, [blobUrl]);

  return (
    <div
      className={`bg-white flex flex-col ${blobUrl ? "" : "min-h-screen"}`}
      style={
        blobUrl
          ? { height: "calc(100vh - 80px)" } // leave room for the site header
          : undefined
      }
    >
      {/* Breadcrumb strip */}
      <div className="border-b border-gray-100 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between gap-4">
          <nav className="text-sm">
            <ol className="flex items-center space-x-2 text-gray-500">
              <li>
                <Link href="/" className="hover:text-green-600">
                  Home
                </Link>
              </li>
              <li>/</li>
              <li>
                <Link href="/tools" className="hover:text-green-600">
                  Tools
                </Link>
              </li>
              <li>/</li>
              <li className="text-gray-900 font-medium">
                3D Gaussian Splatting Viewer
              </li>
            </ol>
          </nav>
          <div className="w-24" />
        </div>
      </div>

      {/* Main area */}
      {!blobUrl ? (
        <div className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`flex-1 min-h-[60vh] rounded-xl border-2 border-dashed flex flex-col items-center justify-center text-center cursor-pointer transition-colors p-8 ${
              isDragging
                ? "border-green-500 bg-green-50"
                : "border-gray-300 bg-gray-50 hover:border-green-400 hover:bg-gray-100"
            }`}
          >
            <svg
              className="h-10 w-10 text-gray-400 mb-3"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 7.5 7.5 12M12 7.5V21"
              />
            </svg>
            <p className="text-gray-700 font-medium mb-1">
              Drop a .ply file here, or click to choose
            </p>
            <p className="text-sm text-gray-500">
              Supported: {ACCEPTED_EXTENSIONS.join(", ")}. Everything runs in
              your browser — nothing is uploaded.
            </p>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_EXTENSIONS.join(",")}
              className="hidden"
              onChange={(e) => acceptFile(e.target.files?.[0])}
            />
            {error && (
              <p className="mt-4 text-sm text-red-600">{error}</p>
            )}
          </div>
        </div>
      ) : (
        <div className="relative flex-1 w-full bg-black">
          {/* File info (top-left) */}
          <div className="absolute top-3 left-3 z-10 bg-black/60 backdrop-blur text-white text-sm rounded px-3 py-1.5 truncate max-w-[55%]">
            {fileName}
          </div>
          {/* Load-another (top-right) */}
          <button
            onClick={reset}
            className="absolute top-3 right-3 z-10 bg-black/60 hover:bg-black/80 backdrop-blur text-white text-sm rounded px-3 py-1.5 transition-colors"
          >
            Load another file
          </button>
          {/* Help hint (bottom-left) */}
          <div className="absolute bottom-3 left-3 z-10 text-white/60 text-xs">
            Drag to orbit · scroll to zoom · right-click-drag to pan
          </div>
          <SplatViewer url={blobUrl} />
        </div>
      )}
    </div>
  );
}
