import { useRef, useState } from "react";
import { deleteDocument, uploadDocument } from "../api/client";
import type { DocumentItem } from "../api/client";

interface Props {
  collection: string;
  documents: DocumentItem[];
  onRefresh: () => void;
}

export default function DocumentUpload({ collection, documents, onRefresh }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await uploadDocument(collection, file);
      onRefresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function handleDelete(source: string) {
    setError(null);
    try {
      await deleteDocument(collection, source);
      onRefresh();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <div className="border-b border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-gray-300">Documents</h2>
        <label
          className={`text-xs px-3 py-1.5 rounded cursor-pointer transition-colors ${
            uploading
              ? "bg-gray-700 text-gray-400 cursor-not-allowed"
              : "bg-indigo-600 hover:bg-indigo-500 text-white"
          }`}
        >
          {uploading ? "Uploading…" : "Upload"}
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt"
            className="hidden"
            onChange={handleFileChange}
            disabled={uploading}
          />
        </label>
      </div>

      {error && <p className="text-xs text-red-400 mb-2">{error}</p>}

      {documents.length === 0 ? (
        <p className="text-xs text-gray-500">No documents yet. Upload a PDF or .txt file.</p>
      ) : (
        <ul className="space-y-1">
          {documents.map((doc) => (
            <li
              key={doc.source}
              className="flex items-center justify-between text-xs text-gray-300 bg-gray-800 rounded px-3 py-1.5 group"
            >
              <span className="truncate">{doc.source}</span>
              <button
                onClick={() => handleDelete(doc.source)}
                className="text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity ml-2 shrink-0"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
