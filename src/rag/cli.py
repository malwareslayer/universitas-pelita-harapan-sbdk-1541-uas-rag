import argparse
from collections.abc import Sequence
from pathlib import Path

import cloudflare

from .app import create
from .ingest import ingestion

parser = argparse.ArgumentParser(prog='rag', description='Policy RAG')
command = parser.add_subparsers(dest='command')


def main(argv: Sequence[str] | None = None) -> int:
  serve = command.add_parser('serve', help='Run FastAPI server')
  serve.add_argument('--cloudflare-account-id', type=str, required=True, help='Cloudflare Account ID')
  serve.add_argument('--cloudflare-api-token', type=str, required=True, help='Cloudflare API Token')
  serve.add_argument('--cloudflare-index-name', type=str, required=True, help='Cloudflare Vectorize Index Name')
  serve.add_argument(
    '--cloudflare-generation-model-name', type=str, default='@cf/google/gemma-3-12b-it', help='Cloudflare Model Name'
  )
  serve.add_argument(
    '--cloudflare-embedding-model-name',
    type=str,
    default='@cf/google/embeddinggemma-300m',
    help='Cloudflare Model Name',
  )
  serve.add_argument('--top-k', type=int, default=5, help='Top-K Vector Search Results')
  serve.add_argument('--host', type=str, default='127.0.0.1', help='FastAPI host')
  serve.add_argument('--port', type=int, default=8080, help='FastAPI port')

  ingest = command.add_parser('ingest', help='Ingest Documents To Cloudflare AI & Vector')
  ingest.add_argument('--cloudflare-account-id', type=str, required=True, help='Cloudflare Account ID')
  ingest.add_argument('--cloudflare-api-token', type=str, required=True, help='Cloudflare API Token')
  ingest.add_argument('--cloudflare-index-name', type=str, required=True, help='Cloudflare Vectorize Index Name')
  ingest.add_argument(
    '--cloudflare-model-name', type=str, default='@cf/google/embeddinggemma-300m', help='Cloudflare Model Name'
  )
  ingest.add_argument(
    '--docs',
    type=Path,
    default=Path(__file__).resolve().parent.parent.parent / 'docs',
    help='Directory Containing Documents',
  )
  ingest.add_argument('--top-k', type=int, default=5, help='Top-K Vector Search Results')
  ingest.add_argument('--chunk-size', type=int, default=2048, help='Chunk')
  ingest.add_argument('--chunk-overlap', type=int, default=512, help='Overlap')

  c = command.add_parser('create', help='Create Cloudflare Vectorize Index')
  c.add_argument('name', type=str, help='Cloudflare Vectorize Index Name To Create')
  c.add_argument('dimension', type=int, help='Cloudflare Vectorize Dimension')
  c.add_argument('--metric', type=str, default='cosine', help='Cloudflare Vectorize Metric Search')
  c.add_argument('--cloudflare-account-id', type=str, required=True, help='Cloudflare Account ID')
  c.add_argument('--cloudflare-api-token', type=str, required=True, help='Cloudflare API Token')

  d = command.add_parser('delete', help='Delete Cloudflare Vectorize Index')
  d.add_argument('name', type=str, help='Cloudflare Vectorize Index Name To Delete')
  d.add_argument('--cloudflare-account-id', type=str, required=True, help='Cloudflare Account ID')
  d.add_argument('--cloudflare-api-token', type=str, required=True, help='Cloudflare API Token')

  args = parser.parse_args(argv)

  match args.command:
    case 'serve':
      create(
        args.cloudflare_account_id,
        args.cloudflare_api_token,
        args.cloudflare_index_name,
        args.cloudflare_generation_model_name,
        args.cloudflare_embedding_model_name,
        args.top_k,
      )
    case 'ingest':
      ingestion(
        account_id=args.cloudflare_account_id,
        api_token=args.cloudflare_api_token,
        index_name=args.cloudflare_index_name,
        model_name=args.cloudflare_model_name,
        documents=args.docs,
        top_k=args.top_k,
        size=args.chunk_size,
        overlap=args.chunk_overlap,
      )
    case 'create':
      cf = cloudflare.Client(api_token=args.cloudflare_api_token)
      # noinspection PyTypeChecker
      cf.vectorize.indexes.create(
        account_id=args.cloudflare_account_id,
        config={
          'dimensions': 768,
          'metric': 'cosine',
        },
        name=args.name,
      )
    case 'delete':
      cf = cloudflare.Client(api_token=args.cloudflare_api_token)
      cf.vectorize.indexes.delete(index_name=args.name, account_id=args.cloudflare_account_id)
    case _:
      raise AttributeError(f'Unknown command: {argv}')

  return 0


__all__ = ['main']
