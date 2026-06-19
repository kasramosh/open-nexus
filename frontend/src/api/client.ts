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
    `/collections/${encodeURIComponent(collection)}/documents/${source}`,
    { method: "DELETE" }
  );

export const queryCollection = (collection: string, query: string, top_k = 5) =>
  request<QueryResponse>(`/collections/${encodeURIComponent(collection)}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });
