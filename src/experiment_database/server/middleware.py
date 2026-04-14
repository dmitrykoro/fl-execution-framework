import hashlib

from datetime import datetime, timezone
from aiohttp import web

from src.experiment_database.session import get_db
from src.experiment_database.models import AuthToken


PUBLIC_ROUTES = {"/", "/experiments", "/experiments/compare", "/experiments/user/{email}"} #I understand if on the server we dont want to keep these public, but for now we have them


def _extract_bearer_token(request: web.Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[len("Bearer "):].strip()
    return token or None


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _is_token_valid(token_hash: str) -> bool:
    """Check token hash exists and is not expired."""

    now = datetime.now(timezone.utc)

    with get_db() as session:
        token_row = (
            session.query(AuthToken)
            .filter(AuthToken.token_hash == token_hash)
            .first()
        )
        if token_row is None:
            return False

        # expires_at should be timezone-aware; compare in UTC
        if token_row.expires_at is not None and token_row.expires_at <= now:
            return False

        return True

def _is_public_route(path: str) -> bool:
    if path in {"/", "/experiments", "/experiments/compare", "/experiments/stats"}:
        return True
    if path.startswith("/experiments/user/"):
        return True
    if path.startswith("/experiments/"): 
        return True
    return False

@web.middleware
async def auth_middleware(request: web.Request, handler):
    if _is_public_route(request.path):
        return await handler(request)
    
    raw = _extract_bearer_token(request)
    if raw is None:
        return web.json_response({"error": "Missing or invalid Authorization header"}, status=401)

    token_hash = _hash_token(raw)

    if not _is_token_valid(token_hash):
        return web.json_response({"error": "Unauthorized"}, status=401)

    return await handler(request)
