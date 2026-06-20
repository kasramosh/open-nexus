const BASE = "http://localhost:8000";

export interface Collection {
  name: string;
}

export interface DocumentItem {
  source: string;
}

export interface SourceChunk {
  source: string;
  page_number: number | null;
  chunk_index: number | null;
  content: string;
}

export interface QueryResponse {
  answer: string;
  sources: SourceChunk[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    // Pydantic validation errors return detail as an array of {msg, loc} objects
    const message = Array.isArray(detail)
      ? detail.map((d: { msg: string }) => d.msg).join("; ")
      : (detail ?? "Request failed");
    throw new Error(message);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const getCollections = () =>
  request<{ collections: Collection[] }>("/collections").then((r) => r.collections);

export const createCollection = (name: string) =>
  request<Collection>("/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });

export const deleteCollection = (name: string) =>
  request<void>(`/collections/${encodeURIComponent(name)}`, { method: "DELETE" });

export const getDocuments = (collection: string) =>
  request<{ documents: DocumentItem[] }>(
    `/collections/${encodeURIComponent(collection)}/documents`
  ).then((r) => r.documents);

export const uploadDocument = (collection: string, file: File) => {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentItem>(
    `/collections/${encodeURIComponent(collection)}/documents`,
    { method: "POST", body: form }
  );
};

export const deleteDocument = (collection: string, source: string) =>
  request<void>(
    `/collections/${encodeURIComponent(collection)}/documents/${encodeURIComponent(source)}`,
    { method: "DELETE" }
  );

export const queryCollection = (collection: string, query: string, top_k = 5) =>
  request<QueryResponse>(`/collections/${encodeURIComponent(collection)}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });

interface StreamHandlers {
  onSources?: (sources: SourceChunk[]) => void;
  onToken?: (token: string) => void;
}

// Streams a cited answer over Server-Sent Events. The backend emits one
// `sources` event followed by many `token` events and a final `done` event.
export async function queryCollectionStream(
  collection: string,
  query: string,
  handlers: StreamHandlers,
  top_k = 5
): Promise<void> {
  const res = await fetch(
    `${BASE}/collections/${encodeURIComponent(collection)}/query/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k }),
    }
  );

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    const message = Array.isArray(detail)
      ? detail.map((d: { msg: string }) => d.msg).join("; ")
      : (detail ?? "Request failed");
    throw new Error(message);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; keep the trailing partial frame.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      const event = JSON.parse(payload);
      if (event.type === "sources") handlers.onSources?.(event.sources);
      else if (event.type === "token") handlers.onToken?.(event.content);
    }
  }
}
