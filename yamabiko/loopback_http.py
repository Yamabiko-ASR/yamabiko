"""HTTP calls to Yamabiko's local services: the bundled llama-server and the
pipeline's /livez.

httpx defaults to ``trust_env=True``. On Windows, urllib's proxy lookup
falls back to the system (registry) proxy and httpx ignores its ``<local>``
bypass, so a request to 127.0.0.1 is sent to the proxy: every transcript
leaves the machine and the local call fails. Loopback targets therefore
never consult proxy settings. A non-loopback target (a user-configured
remote ``translation.endpoint``) keeps honoring the user's proxy, which it
may need in order to be reached at all.

Every call site that talks to those services goes through here, so none
can forget the setting. httpx is imported per call so importing this
module stays cheap for the wizard pages that load it eagerly.
"""
from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import httpx


def _trust_env(url: str) -> bool:
    """False for a loopback host: proxy settings must never apply to it."""
    host = urlsplit(url).hostname
    if host is None:
        return True
    if host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def get(url: str, **kwargs: Any) -> "httpx.Response":
    import httpx
    return httpx.get(url, trust_env=_trust_env(url), **kwargs)


def post(url: str, **kwargs: Any) -> "httpx.Response":
    import httpx
    return httpx.post(url, trust_env=_trust_env(url), **kwargs)


def async_client(url: str, **kwargs: Any) -> "httpx.AsyncClient":
    """AsyncClient for requests to ``url``'s host only: the proxy decision
    is made once, from that host."""
    import httpx
    return httpx.AsyncClient(trust_env=_trust_env(url), **kwargs)
