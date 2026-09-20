"""Windows既定オーディオ出力エンドポイントの動的検出 (Core Audio MMDevice).

sounddeviceの `sd.default.device` はPortAudio初期化時に固定されるため、
イヤホン挿抜でWindowsが既定デバイスを切替えても追従しない。
本モジュールは IMMDeviceEnumerator::GetDefaultAudioEndpoint をctypes COMで直接
呼び、既定出力のフレンドリ名をその都度取得する (既定デバイス変更の検出に使用)。

非Windows・COM失敗時は None を返す (呼出側は現状維持)。
"""

import ctypes
import sys
from ctypes import byref, POINTER, c_void_p, c_ulong
from ctypes import wintypes


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8),
    ]

    def __init__(self, data1, data2, data3, b4):
        super().__init__()
        self.Data1 = data1
        self.Data2 = data2
        self.Data3 = data3
        for i, b in enumerate(b4):
            self.Data4[i] = b


def _guid(data1, data2, data3, b4):
    return GUID(data1, data2, data3, b4)


CLSID_MMDeviceEnumerator = _guid(0xBCDE0395, 0xE52F, 0x467C, [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E])
IID_IMMDeviceEnumerator = _guid(0xA95664D2, 0x9614, 0x4F35, [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6])
IID_IMMDevice = _guid(0xD666063F, 0x1587, 0x4E43, [0x81, 0xF1, 0xB9, 0x48, 0xE8, 0x07, 0x36, 0x3F])
IID_IPropertyStore = _guid(0x886D8EEB, 0x8CF2, 0x4446, [0x8D, 0x02, 0xCD, 0xBA, 0x1D, 0xBD, 0xC9, 0x9C])
class PROPERTYKEY(ctypes.Structure):
    _fields_ = [
        ("fmtid", GUID),
        ("pid", ctypes.c_ulong),
    ]


PKEY_Device_FriendlyName = PROPERTYKEY(_guid(0xA45C254E, 0xDF1C, 0x4EFD, [0x80, 0x20, 0x67, 0xD1, 0x46, 0xA8, 0x50, 0xE0]), 14)

ERender, eMultimedia = 0, 1
CLSCTX_ALL = 0x17
VT_LPWSTR = 31
COINIT_APARTMENTTHREADED = 0x2
S_OK = 0
_HRESULT = ctypes.c_long


def _vtable(obj):
    return ctypes.cast(obj, POINTER(POINTER(ctypes.c_void_p)))[0]


def _release(obj):
    """IUnknown::Release (vtable index 2)"""
    if not obj:
        return
    try:
        fn = ctypes.cast(_vtable(obj)[2], ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p))
        fn(obj)
    except Exception:
        pass


def _win_default_output_name() -> str | None:
    """Windows既定出力エンドポイントのフレンドリ名を返す。失敗時 None。"""
    if sys.platform != "win32":
        return None
    ole32 = ctypes.windll.ole32
    # ctypesは引数型未指定だとc_int扱いで、大きいポインタ値がOverflowErrorになる
    ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, c_ulong,
                                       POINTER(GUID), POINTER(c_void_p)]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [c_void_p]
    ole32.CoTaskMemFree.restype = None
    result = None
    try:
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        pdev = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            byref(CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
            byref(IID_IMMDeviceEnumerator), byref(pdev))
        if hr != S_OK or not pdev:
            return None
        try:
            # IMMDeviceEnumerator::GetDefaultAudioEndpoint(eRender, eMultimedia, &device)
            # vtable[4]: HRESULT (IUnknown*, int, int, IMMDevice**)
            get_default = ctypes.cast(
                _vtable(pdev)[4],
                ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, POINTER(ctypes.c_void_p)))
            hdev = ctypes.c_void_p()
            hr = get_default(pdev, ERender, eMultimedia, byref(hdev))
            if hr != S_OK or not hdev:
                return None
            try:
                # IMMDevice::OpenPropertyStore(STGM_READ, &store)  (vtable[4])
                open_store = ctypes.cast(
                    _vtable(hdev)[4],
                    ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.c_ulong,
                                       POINTER(ctypes.c_void_p)))
                store = ctypes.c_void_p()
                hr = open_store(hdev, 0, byref(store))  # STGM_READ = 0
                if hr != S_OK or not store:
                    return None
                try:
                    # IPropertyStore::GetValue(PKEY_Device_FriendlyName, &pv)
                    # vtable[5]。キーは PROPERTYKEY* (fmtid GUID + pid) で渡す。
                    get_value = ctypes.cast(
                        _vtable(store)[5],
                        ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, POINTER(PROPERTYKEY),
                                           ctypes.c_void_p))
                    class PROPVARIANT(ctypes.Structure):
                        _fields_ = [
                            ("vt", wintypes.USHORT),
                            ("wReserved1", wintypes.USHORT),
                            ("wReserved2", wintypes.USHORT),
                            ("wReserved3", wintypes.USHORT),
                            ("p", ctypes.c_void_p),
                        ]
                    pv = PROPVARIANT()
                    hr = get_value(store, byref(PKEY_Device_FriendlyName), byref(pv))
                    if hr == S_OK and pv.vt == VT_LPWSTR and pv.p:
                        try:
                            result = ctypes.wstring_at(pv.p)
                        finally:
                            ole32.CoTaskMemFree(pv.p)
                finally:
                    _release(store)
            finally:
                _release(hdev)
        finally:
            _release(pdev)
    except Exception:
        return None
    finally:
        try:
            ole32.CoUninitialize()
        except Exception:
            pass
    return result