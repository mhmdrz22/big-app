from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from stressguard.security import decode_access_token


bearer_scheme = HTTPBearer(auto_error=False)


def get_store(request: Request):
    return request.app.state.store


def get_settings(request: Request):
    return request.app.state.settings


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    store=Depends(get_store),
    settings=Depends(get_settings),
):
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing_bearer_token")
    try:
        payload = decode_access_token(credentials.credentials, settings.auth_secret, settings.auth_algorithm)
    except Exception:
        raise HTTPException(status_code=401, detail="invalid_token")
    user = store.get_user_by_id(int(payload["sub"]))
    if not user or not user["is_active"]:
        raise HTTPException(status_code=401, detail="inactive_or_missing_user")
    return user


def require_roles(*allowed_roles: str):
    def dependency(user=Depends(get_current_user)):
        if not any(role in user["roles"] for role in allowed_roles):
            raise HTTPException(status_code=403, detail="insufficient_role")
        return user

    return dependency
