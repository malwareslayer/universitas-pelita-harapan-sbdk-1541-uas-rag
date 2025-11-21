import httpx
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI

from .schema import ChatRequest

fastapi = FastAPI(title='Policy-Based RAG API')
fastapi.state.config = None
fastapi.state.openai = None
fastapi.state.httpx = None
fastapi.state.cloudflare_headers = None
fastapi.state.cloudflare_query_url = None

PROMPT = """
Anda adalah asisten AI yang menjawab pertanyaan berdasarkan kebijakan perusahaan.

ATURAN:
1. Jawab HANYA berdasarkan konteks dokumen di bawah
2. Jika informasi TIDAK ADA dalam konteks, jawab: "Saya tidak menemukan informasi terkait dalam kebijakan perusahaan."
3. Jawab SINGKAT dan LANGSUNG (maksimal 2-3 kalimat)
4. JANGAN menambahkan informasi di luar konteks

Konteks dokumen:
{context}

Pertanyaan: {question}

Jawaban singkat:
"""


@fastapi.on_event('startup')
async def _startup():
  config = getattr(fastapi.state, 'serve_config', None)

  if not config:
    raise AttributeError('Server belum dikonfigurasi melalui CLI.')

  timeout = httpx.Timeout(30, connect=16)

  fastapi.state.httpx = httpx.AsyncClient(timeout=timeout)
  fastapi.state.openai = AsyncOpenAI(api_key=config['openai_api_key'])

  base = (
    f'https://api.cloudflare.com/client/v4/accounts/{config["cloudflare_account_id"]}'
    f'/ai/vectorize/indexes/{config["cloudflare_index"]}'
  )

  fastapi.state.cloudflare_query_url = f'{base}/query'
  fastapi.state.cloudflare_headers = {
    'Authorization': f'Bearer {config["cloudflare_api_token"]}',
    'Content-Type': 'application/json',
  }


@fastapi.on_event('shutdown')
async def _shutdown():
  client = getattr(fastapi.state, 'httpx', None)

  if client:
    await client.aclose()


@fastapi.get('/')
async def health():
  return {'status': 'ok'}


@fastapi.post('/chat')
async def chat(request: ChatRequest):
  config = getattr(fastapi.state, 'serve_config', None)
  http = getattr(fastapi.state, 'http', None)
  openai = getattr(fastapi.state, 'ai_client', None)
  headers = getattr(fastapi.state, 'cloudflare_headers', None)
  query = getattr(fastapi.state, 'cloudflare_query_url', None)

  if not all([config, http, openai, headers, query]):
    raise HTTPException(status_code=500, detail='Server belum dikonfigurasi melalui CLI.')

  try:
    embedding = await openai.embeddings.create(
      model=config['openai_embedding_model'],
      input=request.query,
    )
  except Exception as exc:
    raise HTTPException(status_code=502, detail=f'Gagal membuat embedding: {exc}') from exc

  payload = {
    'vector': embedding.data[0].embedding,
    'topK': config['top_k'],
    'returnValues': False,
    'returnMetadata': True,
  }

  try:
    response = await http.post(query, headers=headers, json=payload)
    response.raise_for_status()
  except httpx.HTTPError as exc:
    raise HTTPException(status_code=502, detail=f'Query Cloudflare gagal: {exc}') from exc

  matches = response.json().get('result', {}).get('matches') or []

  if not matches:
    return {'answer': 'Saya tidak menemukan informasi terkait dalam kebijakan perusahaan.'}

  contexts = []

  for match in matches:
    metadata = match.get('metadata') or {}
    chunk = metadata.get('chunk') or ''
    source = metadata.get('source') or 'tidak diketahui'
    contexts.append(f'Sumber: {source}\n{chunk}'.strip())

  context = '\n---\n'.join(contexts)
  system_prompt = PROMPT.format(context=context, question=request.query)

  try:
    completion = await openai.chat.completions.create(
      model=config['openai_chat_model'],
      messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': request.query},
      ],
      temperature=0.2,
      max_tokens=300,
    )
  except Exception as exc:  # noqa: BLE001
    raise HTTPException(status_code=502, detail=f'OpenAI gagal menjawab: {exc}') from exc

  answer = completion.choices[0].message.content.strip()

  if not answer:
    answer = 'Saya tidak menemukan informasi terkait dalam kebijakan perusahaan.'

  return {'answer': answer, 'sources': [block.split('\n', 1)[0] for block in contexts]}


__all__ = ['fastapi']
