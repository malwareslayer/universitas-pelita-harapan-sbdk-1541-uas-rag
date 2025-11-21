from pathlib import Path
from uuid import uuid4

import httpx
from openai import OpenAI


def ingestion(
  base: Path,
  openai_api_key,
  openai_embedding_model,
  cloudflare_account_id,
  cloudflare_index,
  cloudflare_api_token,
  chunk_size=500,
  chunk_overlap=100,
):
  if not base.exists():
    raise FileNotFoundError(f'Folder tidak ditemukan: {base}')

  docs = []

  for path in base.rglob('*'):
    if path.suffix.lower() in {'.md', '.txt'} and path.is_file():
      docs.append((path, path.read_text(encoding='utf-8')))

  if not docs:
    raise RuntimeError('Tidak ada dokumen .md atau .txt yang ditemukan untuk di-ingest.')

  openai = OpenAI(api_key=openai_api_key)
  http = httpx.Client(timeout=httpx.Timeout(30, connect=10))

  url = (
    f'https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}'
    f'/ai/vectorize/indexes/{cloudflare_index}/upsert'
  )

  headers = {
    'Authorization': f'Bearer {cloudflare_api_token}',
    'Content-Type': 'application/json',
  }

  def chunks(text):
    start = 0
    length = len(text)

    while start < length:
      end = start + chunk_size

      yield text[start:end]

      start = end - chunk_overlap

      if start < 0:
        start = 0

  vectors = []

  for path, content in docs:
    for index, chunk in enumerate(chunks(content)):
      chunk = chunk.strip()

      if not chunk:
        continue

      embedding = openai.embeddings.create(
        model=openai_embedding_model,
        input=chunk,
      )

      vector = embedding.data[0].embedding

      metadata = {
        'source': str(path.relative_to(base)),
        'chunk': chunk,
        'chunk_index': index,
      }

      vectors.append(
        {
          'id': f'{path.stem}-{index}-{uuid4()}'.replace(' ', '-'),
          'values': vector,
          'metadata': metadata,
        }
      )

      if len(vectors) >= 50:
        response = http.post(url, headers=headers, json={'vectors': vectors})
        response.raise_for_status()
        vectors = []

  if vectors:
    response = http.post(url, headers=headers, json={'vectors': vectors})
    response.raise_for_status()

  http.close()


__all__ = ['ingestion']
