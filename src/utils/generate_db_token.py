"""Generate token to access the database. Token is issued for 6 month,"""

import hashlib
import secrets

from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from src.experiment_database.models import AuthToken, Base
from src.experiment_database.session import get_db


def hash_token(raw_token: str) -> str:
    # SHA-256 hex digest => 64 chars
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_raw_token() -> str:
    # URL-safe token; increase bytes for more entropy if you want
    return secrets.token_urlsafe(32)


def main():
    raw = generate_raw_token()
    token_hash = hash_token(raw)

    now = datetime.now(timezone.utc)
    token_row = AuthToken(
        created_at=now,
        expires_at=now + relativedelta(months=5),
        token_hash=token_hash,
    )

    with get_db() as session:
        session.add(token_row)
        session.commit()
        session.refresh(token_row)

    print("TOKEN CREATED")
    print("id:", token_row.id)
    print("expires_at:", token_row.expires_at.isoformat())
    print("\nSAVE THIS TOKEN NOW:")
    print(raw)


if __name__ == "__main__":
    main()
