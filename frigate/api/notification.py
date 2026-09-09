"""Notification apis."""

import logging
import os
from typing import Any

from cryptography.hazmat.primitives import serialization
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from py_vapid import Vapid01, utils

from frigate.api.auth import allow_any_authenticated
from frigate.api.defs.tags import Tags
from frigate.const import CONFIG_DIR
from frigate.models import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.notifications])


@router.get(
    "/notifications/pubkey",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Get VAPID public key",
    description="""Gets the VAPID public key for the notifications.
    Returns the public key or an error if notifications are not enabled.
    """,
)
def get_vapid_pub_key(request: Request):
    config = request.app.frigate_config
    notifications_enabled = config.notifications.enabled
    camera_notifications_enabled = [
        c for c in config.cameras.values() if c.enabled and c.notifications.enabled
    ]
    if not (notifications_enabled or camera_notifications_enabled):
        return JSONResponse(
            content=({"success": False, "message": "Notifications are not enabled."}),
            status_code=400,
        )

    key = Vapid01.from_file(os.path.join(CONFIG_DIR, "notifications.pem"))
    raw_pub = key.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return JSONResponse(content=utils.b64urlencode(raw_pub), status_code=200)


@router.post(
    "/notifications/register",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Register notifications",
    description="""Registers a notifications subscription.
    Returns a success message or an error if the subscription is not provided.
    """,
)
def register_notifications(request: Request, body: dict = None):
    json: dict[str, Any] = body or {}
    sub = json.get("sub")

    if not sub:
        return JSONResponse(
            content={"success": False, "message": "Subscription must be provided."},
            status_code=400,
        )

    # remote-user is injected by the /auth subrequest in nginx (auth_request.conf)
    # and cannot be spoofed by the client. allow_any_authenticated() guarantees it
    # is present unless the request arrived on the trusted internal port.
    username = request.headers.get("remote-user")

    if not username or username == "anonymous":
        return JSONResponse(
            content={
                "success": False,
                "message": "Cannot register notifications for an anonymous user.",
            },
            status_code=400,
        )

    updated = (
        User.update(notification_tokens=User.notification_tokens.append(sub))
        .where(User.username == username)
        .execute()
    )

    if not updated:
        return JSONResponse(
            content={"success": False, "message": "Could not find user."},
            status_code=404,
        )

    return JSONResponse(
        content={"success": True, "message": "Successfully saved token."},
        status_code=200,
    )
