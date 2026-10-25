"""Server request parsing (stdlib only).

Parses the Decisions API request body: ``state`` is a string or a list whose
image parts are png/jpeg data URLs (at most one image) and whose other items
must be strings.
"""
from __future__ import annotations

import base64
import binascii
from typing import Any

_DATA_URL_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/jpg;base64,",
)


def decode_data_url(url: Any) -> bytes:
    """Decode a png/jpeg data URL to raw image bytes (ValueError if invalid)."""
    if not isinstance(url, str):
        raise ValueError("image_url url must be a string")
    for prefix in _DATA_URL_PREFIXES:
        if url.startswith(prefix):
            payload = url[len(prefix):]
            break
    else:
        raise ValueError(
            "image_url url must be a data:image/png;base64,... "
            "or data:image/jpeg;base64,... URL"
        )
    if not payload:
        raise ValueError("image_url data URL has an empty payload")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"image_url is not valid base64: {exc}") from exc
    if not raw:
        raise ValueError("image_url data URL decoded to zero bytes")
    return raw


def parse_state(state: Any) -> tuple[str | list[str], bytes | None]:
    """Split a request state into (text, image_bytes or None).

    A single text becomes a str, several texts become a list of str, and no
    text becomes an empty string. At most one image part is allowed.
    """
    if isinstance(state, str):
        return state, None
    if not isinstance(state, list):
        raise ValueError("state must be a string or a list")
    texts: list[str] = []
    image: bytes | None = None
    for item in state:
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            if item.get("type") != "image_url":
                raise ValueError(
                    "list state items must be strings or "
                    "{'type': 'image_url', ...} objects"
                )
            if image is not None:
                raise ValueError("at most one image per request")
            link = item.get("image_url")
            if not isinstance(link, dict) or "url" not in link:
                raise ValueError("image_url item needs image_url.url")
            image = decode_data_url(link["url"])
            continue
        raise ValueError("list state items must be strings or image parts")
    if len(texts) == 1:
        return texts[0], image
    if not texts:
        return "", image
    return texts, image


def parse_body(body: Any) -> tuple[Any, dict[str, Any]]:
    """Extract (state, questions) from a decoded request JSON body."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in body:
        raise ValueError("request body needs 'state'")
    if "questions" not in body:
        raise ValueError("request body needs 'questions'")
    questions = body["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("'questions' must be a non-empty object")
    return body["state"], questions
