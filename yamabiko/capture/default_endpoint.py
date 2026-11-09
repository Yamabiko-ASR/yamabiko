"""Query Windows Core Audio render endpoints (the OS default, and the active set).

Why not PyAudio: PortAudio (pyaudiowpatch) freezes its device list AND the
default device at ``Pa_Initialize()`` and ships no ``Pa_RefreshDeviceList``
binding (verified: pyaudiowpatch 0.2.12.8 exposes no refresh on PyAudio or in
the ``_portaudiowpatch`` C-ext). So a long-lived PyAudio instance can never
observe a runtime default-output switch -- ``get_default_output_device_info()``
returns the value cached at init forever. (PortAudio docs:
https://github.com/PortAudio/portaudio/wiki/HotPlug)

Windows Core Audio's ``IMMDeviceEnumerator::GetDefaultAudioEndpoint`` always
reflects the CURRENT default, so we query it directly via ``ctypes`` -- pure
stdlib, no ``comtypes`` / ``pycaw`` (which would be new frozen-bundle surface,
hard rule #7). This is a polled read (called ~1 Hz by the capture monitor),
not an ``IMMNotificationClient`` -- a deliberately small surface for a
Phase-C, non-gating detector.

NOTE: switch DETECTION is correct by construction (Core Audio is the OS source
of truth for the default endpoint), but the full real-hardware device-switch
confirmation is owed at release prep -- there is no device-switch tool on the
dev box and we don't mutate the operator's default device in a test. The
plumbing (a live query returns the current endpoint ID) IS exercised by
``tests/capture/test_real_pyaudio.py``.

``active_render_endpoint_names`` lists the active endpoints' friendly names for
the stream-death presence check (plan phase 4).
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import POINTER, byref, c_void_p
from typing import Any, Callable, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, guid: str) -> None:
        super().__init__()
        parts = guid.strip("{}").split("-")
        self.Data1 = int(parts[0], 16)
        self.Data2 = int(parts[1], 16)
        self.Data3 = int(parts[2], 16)
        rest = parts[3] + parts[4]
        for i in range(8):
            self.Data4[i] = int(rest[i * 2:i * 2 + 2], 16)


_CLSID_MMDeviceEnumerator = _GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
_IID_IMMDeviceEnumerator = _GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")

_CLSCTX_ALL = 0x17
_E_RENDER = 0          # EDataFlow.eRender
_E_CONSOLE = 0         # ERole.eConsole (the "default device" role)

# vtable slot indices (IUnknown occupies 0..2 on every interface).
_GET_DEFAULT_AUDIO_ENDPOINT = 4   # IMMDeviceEnumerator
_GET_ID = 5                       # IMMDevice
_RELEASE = 2                      # IUnknown
_ENUM_AUDIO_ENDPOINTS = 3   # IMMDeviceEnumerator
_COLLECTION_GET_COUNT = 3   # IMMDeviceCollection
_COLLECTION_ITEM = 4
_OPEN_PROPERTY_STORE = 4    # IMMDevice
_PROPERTY_GET_VALUE = 5     # IPropertyStore
_DEVICE_STATE_ACTIVE = 0x1
_STGM_READ = 0
_VT_LPWSTR = 31


class _PropertyKey(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]


class _PropVariant(ctypes.Structure):
    # vt + three reserved words (8 bytes), then the 16-byte value union:
    # 24 bytes in all on x64. A VT_LPWSTR holds its string pointer in the
    # union's first slot.
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("reserved1", ctypes.c_ushort),
        ("reserved2", ctypes.c_ushort),
        ("reserved3", ctypes.c_ushort),
        ("value", c_void_p),
        ("pad", c_void_p),
    ]


_PKEY_DEVICE_FRIENDLY_NAME = _PropertyKey(
    _GUID("{a45c254e-df1c-4efd-8020-67d146a850e0}"), 14,
)


def _vtbl_call(this_ptr, index, restype, argtypes, *args):
    """Invoke COM vtable slot ``index`` on the interface pointer ``this_ptr``
    (a ``c_void_p``). ``this`` is passed as the implicit first argument."""
    vtable = ctypes.cast(this_ptr, POINTER(c_void_p))[0]
    func_addr = ctypes.cast(c_void_p(vtable), POINTER(c_void_p))[index]
    prototype = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
    return prototype(func_addr)(this_ptr, *args)


def _with_device_enumerator(
    query: Callable[[c_void_p, Any], T | None], what: str,
) -> T | None:
    """Run ``query(enumerator, ole32)`` against a fresh IMMDeviceEnumerator
    and return its result, or ``None`` when COM or the enumerator is
    unavailable or a call raises (``what`` names the query in the debug log).
    Safe to call repeatedly from a worker thread -- it balances its own
    ``CoInitializeEx`` / ``CoUninitialize`` and releases the enumerator;
    ``query`` releases what it acquires.
    """
    try:
        ole32 = ctypes.windll.ole32
    except OSError:  # pragma: no cover - non-Windows
        return None

    # COINIT_MULTITHREADED = 0. S_OK(0)/S_FALSE(1) -> we incremented the COM
    # refcount and must balance it; RPC_E_CHANGED_MODE(0x80010106) -> the
    # thread already chose a different apartment, don't touch the refcount.
    hr = ole32.CoInitializeEx(None, 0)
    balance_com = hr in (0, 1)
    try:
        enumerator = c_void_p()
        if ole32.CoCreateInstance(
            byref(_CLSID_MMDeviceEnumerator), None, _CLSCTX_ALL,
            byref(_IID_IMMDeviceEnumerator), byref(enumerator),
        ) != 0 or not enumerator:
            return None
        try:
            return query(enumerator, ole32)
        finally:
            _vtbl_call(enumerator, _RELEASE, ctypes.c_ulong, ())
    except OSError as exc:
        _log.debug("%s failed: %s", what, exc)
        return None
    finally:
        if balance_com:
            ole32.CoUninitialize()


def current_default_render_endpoint_id() -> str | None:
    """The Windows endpoint-ID string of the CURRENT default render (output)
    device, or ``None`` if it can't be read (no output device, or any COM
    failure). Safe to call repeatedly from a worker thread -- it balances its
    own ``CoInitializeEx`` / ``CoUninitialize`` and releases every interface.
    """
    return _with_device_enumerator(_default_endpoint_id, "default-endpoint query")


def _default_endpoint_id(enumerator: c_void_p, ole32) -> str | None:
    """IMMDevice::GetId of the default render endpoint (eConsole role)."""
    device = c_void_p()
    # ERROR_NOT_FOUND (0x80070490) here = no default render endpoint
    # (no output devices) -> None.
    if _vtbl_call(
        enumerator, _GET_DEFAULT_AUDIO_ENDPOINT, ctypes.c_long,
        (ctypes.c_int, ctypes.c_int, POINTER(c_void_p)),
        _E_RENDER, _E_CONSOLE, byref(device),
    ) != 0 or not device:
        return None
    try:
        pid = c_void_p()
        if _vtbl_call(
            device, _GET_ID, ctypes.c_long,
            (POINTER(c_void_p),), byref(pid),
        ) != 0 or not pid:
            return None
        try:
            return ctypes.wstring_at(pid)
        finally:
            ole32.CoTaskMemFree(pid)
    finally:
        _vtbl_call(device, _RELEASE, ctypes.c_ulong, ())


def active_render_endpoint_names() -> list[str] | None:
    """Friendly names ("Headphones (USB Audio)") of every ACTIVE render
    endpoint, from a fresh Core Audio enumeration; None when the query
    fails, including when any one endpoint's name can't be read (a list
    missing it would call a present endpoint disconnected). PyAudio lists a
    WASAPI loopback as "<friendly name> [Loopback]" and freezes its list at
    init, so a dead capture stream asks Core Audio whether its endpoint is
    still there (plan lead decision 2). Balances its own COM init and
    releases every interface, like the default-endpoint query above."""
    return _with_device_enumerator(_active_endpoint_names, "render-endpoint enumeration")


def _active_endpoint_names(enumerator: c_void_p, ole32) -> list[str] | None:
    """Friendly names of the ACTIVE render endpoints, all or None."""
    collection = c_void_p()
    if _vtbl_call(
        enumerator, _ENUM_AUDIO_ENDPOINTS, ctypes.c_long,
        (ctypes.c_int, ctypes.c_uint32, POINTER(c_void_p)),
        _E_RENDER, _DEVICE_STATE_ACTIVE, byref(collection),
    ) != 0 or not collection:
        return None
    try:
        count = ctypes.c_uint32()
        if _vtbl_call(
            collection, _COLLECTION_GET_COUNT, ctypes.c_long,
            (POINTER(ctypes.c_uint32),), byref(count),
        ) != 0:
            return None
        names = []
        for index in range(count.value):
            name = _friendly_name(collection, index, ole32)
            if not name:
                return None
            names.append(name)
        return names
    finally:
        _vtbl_call(collection, _RELEASE, ctypes.c_ulong, ())


def _friendly_name(collection: c_void_p, index: int, ole32) -> str | None:
    """PKEY_Device_FriendlyName of device ``index`` in ``collection``."""
    device = c_void_p()
    if _vtbl_call(
        collection, _COLLECTION_ITEM, ctypes.c_long,
        (ctypes.c_uint32, POINTER(c_void_p)), index, byref(device),
    ) != 0 or not device:
        return None
    try:
        store = c_void_p()
        if _vtbl_call(
            device, _OPEN_PROPERTY_STORE, ctypes.c_long,
            (ctypes.c_uint32, POINTER(c_void_p)), _STGM_READ, byref(store),
        ) != 0 or not store:
            return None
        try:
            value = _PropVariant()
            if _vtbl_call(
                store, _PROPERTY_GET_VALUE, ctypes.c_long,
                (POINTER(_PropertyKey), POINTER(_PropVariant)),
                byref(_PKEY_DEVICE_FRIENDLY_NAME), byref(value),
            ) != 0:
                return None
            try:
                if value.vt == _VT_LPWSTR and value.value:
                    return ctypes.wstring_at(value.value)
                return None
            finally:
                ole32.PropVariantClear(byref(value))
        finally:
            _vtbl_call(store, _RELEASE, ctypes.c_ulong, ())
    finally:
        _vtbl_call(device, _RELEASE, ctypes.c_ulong, ())
