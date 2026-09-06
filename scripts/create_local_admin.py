from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stressguard.data_store import DataStore
from stressguard.settings import load_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--name", default="Local Admin")
    args = parser.parse_args()

    settings = load_settings(PROJECT_ROOT)
    store = DataStore(settings.database_url)
    store.init_db()

    # Upsert: if this email already exists, its password is RESET to the one given here,
    # so this script always leaves you with working credentials for local use.
    user = store.upsert_admin(email=args.email, full_name=args.name, password=args.password)
    print(f"Admin ready: {user['email']} with roles {', '.join(user['roles'])} (password set to the one you passed)")


if __name__ == "__main__":
    main()
