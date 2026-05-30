CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    category    TEXT NOT NULL,
    content_sha TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS documents_category_idx ON documents (category);

CREATE TABLE IF NOT EXISTS support_gaps (
    id           BIGSERIAL PRIMARY KEY,
    question     TEXT NOT NULL,
    session_id   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    sources       JSONB NOT NULL DEFAULT '[]'::jsonb,
    handoff       BOOLEAN NOT NULL DEFAULT FALSE,
    booking_ref   TEXT,
    booking_email TEXT,
    booking_found BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_session_idx ON conversations (session_id);
CREATE INDEX IF NOT EXISTS conversations_created_idx ON conversations (created_at DESC);
