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

IID_IAudioEndpointVolume = _guid(0x5CDF2C82, 0x841E, 0x4546, [0x97, 0x22, 0x0C, 0xF7, 0x40, 0x78, 0x22, 0x9A])
IID_IAudioSessionManager2 = _guid(0x77AA99A0, 0x1BD6, 0x484F, [0x8B, 0xC7, 0x2C, 0x65, 0x4C, 0x9A, 0x9B, 0x6F])
IID_IAudioSessionControl2 = _guid(0xBFB7FF88, 0x7239, 0x4FC9, [0x8F, 0xA2, 0x07, 0xC9, 0x50, 0xBE, 0x9C, 0x6D])
IID_ISimpleAudioVolume = _guid(0x87CE5498, 0x68D6, 0x44E5, [0x92, 0x15, 0x6D, 0xA4, 0x7E, 0xF8, 0x83, 0xD8])
IID_IAudioMeterInformation = _guid(0xC02216F6, 0xCA67, 0x4B5B, [0x9D, 0x00, 0xD0, 0x08, 0xE7, 0x3E, 0x00, 0x64])

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


def _win_enumerate_render_endpoints():
    """Windowsの全レンダー(出力)エンドポイントを列挙し
    [(friendly_name, state, device_id), ...] を返す (診断用)。
    state: 1=ACTIVE, 2=DISABLED, 4=NOTPRESENT, 8=UNPLUGGED"""
    if sys.platform != "win32":
        return []
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, c_ulong,
                                       POINTER(GUID), POINTER(c_void_p)]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    ole32.CoTaskMemFree.argtypes = [c_void_p]
    ole32.CoTaskMemFree.restype = None
    out = []
    try:
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        pdev = c_void_p()
        hr = ole32.CoCreateInstance(byref(CLSID_MMDeviceEnumerator), None,
                                    CLSCTX_ALL, byref(IID_IMMDeviceEnumerator),
                                    byref(pdev))
        if hr != S_OK or not pdev:
            return []
        try:
            # IMMDeviceEnumerator::EnumAudioEndpoints(eRender, DEVICE_STATE_ALL=0xF, &coll)
            enum_ep = ctypes.cast(
                _vtable(pdev)[3],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, ctypes.c_int, ctypes.c_ulong,
                                   POINTER(c_void_p)))
            coll = c_void_p()
            if enum_ep(pdev, ERender, 0xF, byref(coll)) != S_OK or not coll:
                return []
            try:
                # IMMDeviceCollection::GetCount (vtable[3]) / Item (vtable[4])
                get_count = ctypes.cast(
                    _vtable(coll)[3],
                    ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_uint)))
                cnt = ctypes.c_uint(0)
                get_count(coll, byref(cnt))
                item = ctypes.cast(
                    _vtable(coll)[4],
                    ctypes.WINFUNCTYPE(_HRESULT, c_void_p, ctypes.c_uint,
                                       POINTER(c_void_p)))
                for i in range(cnt.value):
                    d = c_void_p()
                    if item(coll, i, byref(d)) != S_OK or not d:
                        continue
                    try:
                        name = _device_friendly_name(d)
                        # IMMDevice::GetId (vtable[5]) -> LPWSTR
                        get_id = ctypes.cast(
                            _vtable(d)[5],
                            ctypes.WINFUNCTYPE(_HRESULT, c_void_p,
                                               POINTER(ctypes.c_wchar_p)))
                        pid = ctypes.c_wchar_p()
                        dev_id = ""
                        if get_id(d, byref(pid)) == S_OK and pid.value:
                            dev_id = pid.value
                            ole32.CoTaskMemFree(pid)
                        # IMMDevice::GetState (vtable[6]) -> DWORD
                        get_state = ctypes.cast(
                            _vtable(d)[6],
                            ctypes.WINFUNCTYPE(_HRESULT, c_void_p,
                                               POINTER(ctypes.c_ulong)))
                        st = ctypes.c_ulong(0)
                        get_state(d, byref(st))
                        out.append((name, int(st.value), dev_id))
                    finally:
                        _release(d)
            finally:
                _release(coll)
        finally:
            _release(pdev)
    except Exception:
        return out
    finally:
        try:
            ole32.CoUninitialize()
        except Exception:
            pass
    return out


def _device_friendly_name(hdev) -> str | None:
    """IMMDeviceのフレンドリ名を返す (OpenPropertyStore経由)。"""
    try:
        open_store = ctypes.cast(
            _vtable(hdev)[4],
            ctypes.WINFUNCTYPE(_HRESULT, c_void_p, ctypes.c_ulong,
                               POINTER(ctypes.c_void_p)))
        store = c_void_p()
        if open_store(hdev, 0, byref(store)) != S_OK or not store:
            return None
        try:
            get_value = ctypes.cast(
                _vtable(store)[5],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(PROPERTYKEY),
                                   ctypes.c_void_p))
            class PROPVARIANT(ctypes.Structure):
                _fields_ = [
                    ("vt", wintypes.USHORT),
                    ("wReserved1", wintypes.USHORT),
                    ("wReserved2", wintypes.USHORT),
                    ("wReserved3", wintypes.USHORT),
                    ("p", c_void_p),
                ]
            pv = PROPVARIANT()
            if get_value(store, byref(PKEY_Device_FriendlyName), byref(pv)) == S_OK \
                    and pv.vt == VT_LPWSTR and pv.p:
                try:
                    return ctypes.wstring_at(pv.p)
                finally:
                    ctypes.windll.ole32.CoTaskMemFree(pv.p)
            return None
        finally:
            _release(store)
    except Exception:
        return None


def _win_default_output_peak():
    """Windows既定出力エンドポイントの現在ピーク値 (0.0-1.0) を返す (診断用)。
    失敗時 None。実際に信号がエンドポイントへ届いているかの判定に使う。"""
    if sys.platform != "win32":
        return None
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, c_ulong,
                                       POINTER(GUID), POINTER(c_void_p)]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    peak = None
    try:
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        pdev = c_void_p()
        if ole32.CoCreateInstance(byref(CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
                                  byref(IID_IMMDeviceEnumerator), byref(pdev)) != S_OK or not pdev:
            return None
        try:
            get_default = ctypes.cast(
                _vtable(pdev)[4],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, ctypes.c_int,
                                   ctypes.c_int, POINTER(c_void_p)))
            hdev = c_void_p()
            if get_default(pdev, ERender, eMultimedia, byref(hdev)) != S_OK or not hdev:
                return None
            try:
                activate = ctypes.cast(
                    _vtable(hdev)[3],
                    ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(GUID),
                                       ctypes.c_ulong, c_void_p, POINTER(c_void_p)))
                meter = c_void_p()
                if activate(hdev, byref(IID_IAudioMeterInformation), CLSCTX_ALL,
                            None, byref(meter)) == S_OK and meter:
                    try:
                        # IAudioMeterInformation::GetPeakValue (vtable[3])
                        get_peak = ctypes.cast(
                            _vtable(meter)[3],
                            ctypes.WINFUNCTYPE(_HRESULT, c_void_p,
                                               POINTER(ctypes.c_float)))
                        p = ctypes.c_float(-1.0)
                        if get_peak(meter, byref(p)) == S_OK:
                            peak = float(p.value)
                    finally:
                        _release(meter)
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
    return peak


def _win_session_volumes():
    """既定出力の全オーディオセッション (アプリ別ミキサー) を列挙し
    [(pid, volume, muted), ...] を返す (診断用)。"""
    if sys.platform != "win32":
        return []
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, c_ulong,
                                       POINTER(GUID), POINTER(c_void_p)]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    out = []
    enum = None
    hdev = None
    try:
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        pdev = c_void_p()
        if ole32.CoCreateInstance(byref(CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
                                  byref(IID_IMMDeviceEnumerator), byref(pdev)) != S_OK or not pdev:
            return []
        try:
            get_default = ctypes.cast(
                _vtable(pdev)[4],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, ctypes.c_int,
                                   ctypes.c_int, POINTER(c_void_p)))
            hdev = c_void_p()
            if get_default(pdev, ERender, eMultimedia, byref(hdev)) != S_OK or not hdev:
                return []
            # IMMDevice::Activate(IAudioSessionManager2)
            activate = ctypes.cast(
                _vtable(hdev)[3],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(GUID),
                                   ctypes.c_ulong, c_void_p, POINTER(c_void_p)))
            mgr = c_void_p()
            if activate(hdev, byref(IID_IAudioSessionManager2), CLSCTX_ALL,
                        None, byref(mgr)) != S_OK or not mgr:
                return []
            try:
                # IAudioSessionManager2::GetSessionEnumerator (vtable[5])
                get_enum = ctypes.cast(
                    _vtable(mgr)[5],
                    ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(c_void_p)))
                enum = c_void_p()
                if get_enum(mgr, byref(enum)) != S_OK or not enum:
                    return []
                # IAudioSessionEnumerator::GetCount (vtable[3]) / GetSession (vtable[4])
                get_count = ctypes.cast(
                    _vtable(enum)[3],
                    ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_int)))
                get_session = ctypes.cast(
                    _vtable(enum)[4],
                    ctypes.WINFUNCTYPE(_HRESULT, c_void_p, ctypes.c_int,
                                       POINTER(c_void_p)))
                cnt = ctypes.c_int(0)
                get_count(enum, byref(cnt))
                for i in range(cnt.value):
                    ctrl = c_void_p()
                    if get_session(enum, i, byref(ctrl)) != S_OK or not ctrl:
                        continue
                    try:
                        pid, vol, muted, state = _session_info(ctrl)
                        out.append((pid, vol, muted, state))
                    finally:
                        _release(ctrl)
            finally:
                _release(mgr)
        finally:
            _release(pdev)
    except Exception:
        return out
    finally:
        try:
            if enum:
                _release(enum)
            if hdev:
                _release(hdev)
        except Exception:
            pass
        try:
            ole32.CoUninitialize()
        except Exception:
            pass
    return out


def _session_info(ctrl):
    """IAudioSessionControl → (pid, volume, muted, state)。取得不能項目はNone。
    state: 0=Inactive, 1=Active, 2=Expired"""
    pid = None
    vol = None
    muted = None
    state = None
    # IAudioSessionControl::GetState (vtable[3])
    try:
        get_state = ctypes.cast(
            _vtable(ctrl)[3],
            ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_int)))
        st = ctypes.c_int(-1)
        if get_state(ctrl, byref(st)) == S_OK:
            state = int(st.value)
    except Exception:
        pass
    # QueryInterface で IAudioSessionControl2 (pid) と ISimpleAudioVolume
    qi = ctypes.cast(_vtable(ctrl)[0],
                     ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(GUID),
                                        POINTER(c_void_p)))
    c2 = c_void_p()
    if qi(ctrl, byref(IID_IAudioSessionControl2), byref(c2)) == S_OK and c2:
        try:
            # IAudioSessionControl2::GetProcessId (vtable[14])
            get_pid = ctypes.cast(
                _vtable(c2)[14],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_ulong)))
            p = ctypes.c_ulong(0)
            if get_pid(c2, byref(p)) == S_OK:
                pid = int(p.value)
        finally:
            _release(c2)
    sv = c_void_p()
    if qi(ctrl, byref(IID_ISimpleAudioVolume), byref(sv)) == S_OK and sv:
        try:
            # ISimpleAudioVolume: GetMasterVolume (vtable[4]), GetMute (vtable[6])
            get_vol = ctypes.cast(
                _vtable(sv)[4],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_float)))
            v = ctypes.c_float(-1.0)
            if get_vol(sv, byref(v)) == S_OK:
                vol = float(v.value)
            get_mute = ctypes.cast(
                _vtable(sv)[6],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_int)))
            m = ctypes.c_int(-1)
            if get_mute(sv, byref(m)) == S_OK:
                muted = bool(m.value)
        finally:
            _release(sv)
    return (pid, vol, muted, state)


def _win_default_output_mute_volume():
    """Windows既定出力エンドポイントの (muted, volume 0.0-1.0) を返す。
    取得失敗時は (None, None)。デバッグ/診断用。"""
    if sys.platform != "win32":
        return (None, None)
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.argtypes = [c_void_p, c_ulong]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoCreateInstance.argtypes = [POINTER(GUID), c_void_p, c_ulong,
                                       POINTER(GUID), POINTER(c_void_p)]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    muted = None
    volume = None
    try:
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        pdev = c_void_p()
        hr = ole32.CoCreateInstance(
            byref(CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
            byref(IID_IMMDeviceEnumerator), byref(pdev))
        if hr != S_OK or not pdev:
            return (None, None)
        try:
            get_default = ctypes.cast(
                _vtable(pdev)[4],
                ctypes.WINFUNCTYPE(_HRESULT, c_void_p, ctypes.c_int,
                                   ctypes.c_int, POINTER(c_void_p)))
            hdev = c_void_p()
            hr = get_default(pdev, ERender, eMultimedia, byref(hdev))
            if hr != S_OK or not hdev:
                return (None, None)
            try:
                # IMMDevice::Activate(IID_IAudioEndpointVolume, CLSCTX_ALL, None, &vol)
                activate = ctypes.cast(
                    _vtable(hdev)[3],
                    ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(GUID),
                                       ctypes.c_ulong, c_void_p,
                                       POINTER(c_void_p)))
                vol = c_void_p()
                hr = activate(hdev, byref(IID_IAudioEndpointVolume), CLSCTX_ALL,
                              None, byref(vol))
                if hr == S_OK and vol:
                    try:
                        # vtable[15] GetMute(BOOL*), vtable[9] GetMasterVolumeLevelScalar(float*)
                        get_mute = ctypes.cast(
                            _vtable(vol)[15],
                            ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_int)))
                        m = ctypes.c_int(-1)
                        if get_mute(vol, byref(m)) == S_OK:
                            muted = bool(m.value)
                        get_vol = ctypes.cast(
                            _vtable(vol)[9],
                            ctypes.WINFUNCTYPE(_HRESULT, c_void_p, POINTER(ctypes.c_float)))
                        v = ctypes.c_float(-1.0)
                        if get_vol(vol, byref(v)) == S_OK:
                            volume = float(v.value)
                    finally:
                        _release(vol)
            finally:
                _release(hdev)
        finally:
            _release(pdev)
    except Exception:
        return (None, None)
    finally:
        try:
            ole32.CoUninitialize()
        except Exception:
            pass
    return (muted, volume)


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