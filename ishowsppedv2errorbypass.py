# -*- coding: utf-8 -*-
"""
HYPER-OPTIMIZED Aksharamukha Transliterator — Crash-proof single-file app
- Local binding (aksharamukha) if installed (ProcessPoolExecutor).
- Async API fallback with httpx and connection pooling.
- Disk-backed temp outputs, chunking, caching.
- Safe execution wrapper: logs errors and continues (no crash).
"""
import os
import io
import sys
import math
import time
import json
import hashlib
import tempfile
import shutil
import zipfile
import asyncio
import traceback
import gc
import re
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List

import streamlit as st

# Optional performance libs
try:
    import orjson as _orjson
    def dump_bytes(obj):
        return _orjson.dumps(obj)
    def loads_bytes(b):
        return _orjson.loads(b)
except Exception:
    _orjson = None
    def dump_bytes(obj):
        return json.dumps(obj).encode("utf-8")
    def loads_bytes(b):
        return json.loads(b)

# HTTP client
try:
    import httpx
except Exception:
    httpx = None

# Local binding detection
try:
    from aksharamukha import transliterate as ak_trans
    LOCAL_BINDING = True
except Exception:
    LOCAL_BINDING = False

# ----------------- FULL SCRIPT MAP -----------------
SCRIPT_MAP = {
    "Ahom": "Ahom", "Arabic": "Arab", "Ariyaka": "Ariyaka", "Assamese": "Assamese",
    "Avestan": "Avestan", "Balinese": "Balinese", "Batak Karo": "BatakKaro",
    "Batak Mandailing": "BatakManda", "Batak Pakpak": "BatakPakpak",
    "Batak Simalungun": "BatakSima", "Batak Toba": "BatakToba",
    "Bengali (Bangla)": "Bengali", "Bhaiksuki": "Bhaiksuki", "Brahmi": "Brahmi",
    "Buginese (Lontara)": "Buginese", "Buhid": "Buhid", "Burmese (Myanmar)": "Burmese",
    "Chakma": "Chakma", "Cham": "Cham", "Cyrillic (Russian)": "RussianCyrillic",
    "Devanagari": "Devanagari", "Dives Akuru": "DivesAkuru", "Dogra": "Dogra",
    "Elymaic": "Elym", "Ethiopic (Abjad)": "Ethi", "Gondi (Gunjala)": "GunjalaGondi",
    "Gondi (Masaram)": "MasaramGondi", "Grantha": "Grantha", "Grantha (Pandya)": "GranthaPandya",
    "Gujarati": "Gujarati", "Hanunoo": "Hanunoo", "Hatran": "Hatr", "Hebrew": "Hebrew",
    "Hebrew (Judeo-Arabic)": "Hebr-Ar", "Imperial Aramaic": "Armi",
    "Inscriptional Pahlavi": "Phli", "Inscriptional Parthian": "Prti",
    "Japanese (Hiragana)": "Hiragana", "Japanese (Katakana)": "Katakana",
    "Javanese": "Javanese", "Kaithi": "Kaithi", "Kannada": "Kannada", "Kawi": "Kawi",
    "Khamti Shan": "KhamtiShan", "Kharoshthi": "Kharoshthi", "Khmer (Cambodian)": "Khmer",
    "Khojki": "Khojki", "Khom Thai": "KhomThai", "Khudawadi": "Khudawadi", "Lao": "Lao",
    "Lao (Pali)": "LaoPali", "Lepcha": "Lepcha", "Limbu": "Limbu", "Mahajani": "Mahajani",
    "Makasar": "Makasar", "Malayalam": "Malayalam", "Manichaean": "Mani",
    "Marchen": "Marchen", "Meetei Mayek (Manipuri)": "MeeteiMayek", "Modi": "Modi",
    "Mon": "Mon", "Mongolian (Ali Gali)": "Mongolian", "Mro": "Mro", "Multani": "Multani",
    "Nabataean": "Nbat", "Nandinagari": "Nandinagari", "Newa (Nepal Bhasa)": "Newa",
    "Old North Arabian": "Narb", "Old Persian": "OldPersian", "Old Sogdian": "Sogo",
    "Old South Arabian": "Sarb", "Oriya (Odia)": "Oriya", "Pallava": "Pallava",
    "Palmyrene": "Palm", "Persian": "Arab-Fa", "PhagsPa": "PhagsPa", "Phoenician": "Phnx",
    "Psalter Pahlavi": "Phlp", "Punjabi (Gurmukhi)": "Gurmukhi", "Ranjana (Lantsa)": "Ranjana",
    "Rejang": "Rejang", "Rohingya (Hanifi)": "HanifiRohingya", "Roman (Baraha North)": "BarahaNorth",
    "Roman (Baraha South)": "BarahaSouth", "Roman (Colloquial)": "RomanColloquial",
    "Roman (DMG Persian)": "PersianDMG", "Roman (Harvard-Kyoto)": "HK", "Roman (IAST)": "IAST",
    "Roman (IAST: Pāḷi)": "IASTPali", "Roman (IPA Indic)": "IPA", "Roman (ISO 15919 Indic)": "ISO",
    "Roman (ISO 15919: Pāḷi)": "ISOPali", "Roman (ISO 233 Arabic)": "ISO233",
    "Roman (ISO 259 Hebrew)": "ISO259", "Roman (ITRANS)": "Itrans",
    "Roman (Library of Congress)": "RomanLoC", "Roman (Readable)": "RomanReadable",
    "Roman (SBL Hebrew)": "HebrewSBL", "Roman (SLP1)": "SLP1",
    "Roman (Semitic Typeable)": "Type", "Roman (Semitic)": "Latn", "Roman (Titus)": "Titus",
    "Roman (Velthuis)": "Velthuis", "Roman (WX)": "WX", "Samaritan": "Samr",
    "Santali (Ol Chiki)": "Santali", "Saurashtra": "Saurashtra", "Shahmukhi": "Shahmukhi",
    "Shan": "Shan", "Sharada": "Sharada", "Siddham": "Siddham", "Sinhala": "Sinhala",
    "Sogdian": "Sogd", "Sora Sompeng": "SoraSompeng", "Soyombo": "Soyombo",
    "Sundanese": "Sundanese", "Syloti Nagari": "SylotiNagri", "Syriac (Eastern)": "Syrn",
    "Syriac (Estrangela)": "Syre", "Syriac (Western)": "Syrj", "Tagalog": "Tagalog",
    "Tagbanwa": "Tagbanwa", "Tai Laing": "TaiLaing", "Takri": "Takri", "Tamil": "Tamil",
    "Tamil (Extended)": "TamilExtended", "Tamil Brahmi": "TamilBrahmi", "Telugu": "Telugu",
    "Thaana (Dhivehi)": "Thaana", "Thai": "Thai", "Tham (Lanna)": "TaiTham",
    "Tham (Lao)": "LaoTham", "Tham (Tai Khuen)": "KhuenTham", "Tham (Tai Lue)": "LueTham",
    "Tibetan": "Tibetan", "Tirhuta (Maithili)": "Tirhuta", "Ugaritic": "Ugar",
    "Urdu": "Urdu", "Vatteluttu": "Vatteluttu", "Wancho": "Wancho",
    "Warang Citi": "WarangCiti", "Zanabazar Square": "ZanabazarSquare",
}

# ----------------- Config / Constants -----------------
API_ENDPOINT = "https://aksharamukha.appspot.com/api/transliterate"
TEMP_BASE = Path(tempfile.gettempdir())
RUN_ID = f"{os.getpid()}_{int(time.time())}"
CACHE_DIR = TEMP_BASE / f"aksha_cache_{RUN_ID}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default concurrency
DEFAULT_IO_WORKERS = max(4, (os.cpu_count() or 4))
DEFAULT_BATCH_SIZE = 50
DEFAULT_CHUNK_MB = 4
DEFAULT_API_CONCURRENCY = 12

# Process pool for local binding
LOCAL_POOL = None
if LOCAL_BINDING:
    pool_workers = max(1, (os.cpu_count() or 2) // 2)
    LOCAL_POOL = ProcessPoolExecutor(max_workers=pool_workers)

# ----------------- Safety helpers -----------------
INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

def safe_folder(name: str) -> str:
    """Sanitize folder name for Windows and other OSes."""
    if not isinstance(name, str):
        name = str(name)
    # replace invalid chars
    s = INVALID_CHARS_RE.sub("_", name)
    # also replace sequences of spaces/dots at end
    s = s.strip(" .")
    # limit length to reasonable size
    if len(s) == 0:
        s = "untitled"
    return s[:200]

def log_error(log_path: Path, task_name: str, exc: Exception):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n==== ERROR TASK: {task_name} ====\n")
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(str(exc) + "\n")
            f.write(traceback.format_exc())
    except Exception:
        # Last-resort: print to stderr
        print("Failed to write log:", file=sys.stderr)
        traceback.print_exc()

async def async_safe_exec(task_name: str, coro, log_path: Path):
    """Run coroutine, catch exceptions, log, and return None on error."""
    try:
        return await coro
    except Exception as e:
        log_error(log_path, task_name, e)
        return None

def sync_safe_exec(task_name: str, func, log_path: Path, *args, **kwargs):
    """Run sync function safely, log exceptions, return None on error."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_error(log_path, task_name, e)
        return None

# ----------------- Cache helpers -----------------
def sha1_key(source: str, target: str, preview: str) -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(target.encode("utf-8"))
    h.update(b"\x00")
    h.update(preview.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def cache_read(key: str):
    p = CACHE_DIR / key
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

def cache_write(key: str, text: str):
    try:
        (CACHE_DIR / key).write_text(text, encoding="utf-8")
    except Exception:
        pass

# ----------------- I/O & chunking -----------------
def read_file_fast(f):
    try:
        raw = f.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return raw
    except Exception:
        return ""
    finally:
        try:
            f.close()
        except Exception:
            pass

def chunk_text_by_bytes(s: str, max_bytes: int) -> List[str]:
    b = s.encode("utf-8")
    L = len(b)
    if L <= max_bytes:
        return [s]
    out = []
    start = 0
    while start < L:
        end = min(start + max_bytes, L)
        # don't cut multibyte char
        while end > start and (b[end - 1] & 0xC0) == 0x80:
            end -= 1
        out.append(b[start:end].decode("utf-8", errors="ignore"))
        start = end
    return out

def write_file_safe(path: Path, content: str, log_path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        # attempt recovery: write best-effort bytes
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8", errors="ignore"))
            return True
        except Exception as ee:
            log_error(log_path, f"Write file {path}", ee)
            return False

# ----------------- Backends -----------------
def _local_transform(src: str, tgt: str, text: str, options=None) -> str:
    try:
        if options:
            return ak_trans.process(src, tgt, text, **options)
        else:
            return ak_trans.process(src, tgt, text)
    except Exception:
        # return original text on error (so we still have something)
        return text

async def _api_transform_async(client: "httpx.AsyncClient", src: str, tgt: str, text: str, timeout=180):
    payload = {"source": src, "target": tgt, "text": text}
    try:
        r = await client.post(API_ENDPOINT, json=payload, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json().get("text", r.text)
            except Exception:
                return r.text
        else:
            return r.text
    except Exception:
        return text

# ----------------- High-level convert per-file (safe usage implemented in orchestration) -----------------
async def convert_text_async(source_id: str, target_id: str, text: str, client: "httpx.AsyncClient", use_local: bool, max_chunk_bytes: int, api_semaphore: asyncio.Semaphore=None, options=None):
    preview = text[:4096]
    key = sha1_key(source_id, target_id, preview)
    cached = cache_read(key)
    if cached:
        return cached

    if use_local:
        chunks = chunk_text_by_bytes(text, max_chunk_bytes)
        if len(chunks) == 1:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(LOCAL_POOL, partial(_local_transform, source_id, target_id, chunks[0], options))
            cache_write(key, res)
            return res
        else:
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(LOCAL_POOL, partial(_local_transform, source_id, target_id, c, options)) for c in chunks]
            pieces = await asyncio.gather(*tasks)
            out = "".join(pieces)
            cache_write(key, out)
            return out
    else:
        chunks = chunk_text_by_bytes(text, max_chunk_bytes)
        if len(chunks) == 1:
            out = await _api_transform_async(client, source_id, target_id, text)
            cache_write(key, out)
            return out
        else:
            sem = api_semaphore or asyncio.Semaphore(DEFAULT_API_CONCURRENCY)
            async def guarded(chunk):
                async with sem:
                    return await _api_transform_async(client, source_id, target_id, chunk)
            tasks = [guarded(c) for c in chunks]
            pieces = await asyncio.gather(*tasks)
            out = "".join(pieces)
            cache_write(key, out)
            return out

# ----------------- Streamlit UI & orchestration -----------------
st.set_page_config(page_title="⚡ HYPER-OPTIMIZED Aksharamukha Transliterator (SAFE)", layout="wide")
st.title("⚡ HYPER-OPTIMIZED Aksharamukha Transliterator — Crash-Proof")

if LOCAL_BINDING:
    st.success("🚀 Local binding active (aksharamukha)")
else:
    st.info("📡 API mode (httpx). For fastest runs install: pip install aksharamukha")

# Upload
uploaded_files = st.file_uploader("📤 Upload .txt files (many allowed)", type="txt", accept_multiple_files=True)

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    source_script = st.selectbox("Source Script", list(SCRIPT_MAP.keys()), index=0)
with col2:
    mode = st.radio("Conversion Mode", ["Single Target", "All Scripts"], index=0)
with col3:
    target_script = st.selectbox("Target Script", list(SCRIPT_MAP.keys()), index=10)

col4, col5 = st.columns(2)
with col4:
    workers = st.slider("I/O worker threads", 2, max(4, (os.cpu_count() or 4) * 4), DEFAULT_IO_WORKERS)
with col5:
    batch_size = st.slider("Batch Size (files per batch)", 5, 500, DEFAULT_BATCH_SIZE)

col6, col7 = st.columns(2)
with col6:
    chunk_mb = st.slider("Text chunk size (MB)", 1, 64, DEFAULT_CHUNK_MB)
with col7:
    api_concurrency = st.slider("API concurrency (if using API mode)", 2, 64, DEFAULT_API_CONCURRENCY)

compress = st.checkbox("🗜️ Compress ZIP (smaller file, slower)", False)
strip_unicode = st.checkbox("🧹 Normalize text (NFC)", True)
disable_spellcheck = st.checkbox("⚡ Disable extra aksharamukha options (recommended)", True)

# Options to pass to local binding if used
local_options = {}
if disable_spellcheck:
    local_options = {"pre_options": ["SanskritMode"], "post_options": ["DoNotReplacePunctuation"]}

# Action button
if st.button("⚡ START HYPER-FAST CRASH-PROOF CONVERSION"):
    if not uploaded_files:
        st.error("❌ Please upload at least one file")
        st.stop()

    # Setup per-run output and log
    OUT_DIR = TEMP_BASE / f"aksha_out_{RUN_ID}"
    LOG_PATH = OUT_DIR / "errors.log"
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    sync_safe_exec("Create out dir", lambda: OUT_DIR.mkdir(parents=True, exist_ok=True), LOG_PATH)

    files = uploaded_files[:10000]
    n_files = len(files)
    st.info(f"Reading {n_files} files using {workers} threads...")

    with ThreadPoolExecutor(max_workers=workers) as tpe:
        contents = list(tpe.map(read_file_fast, files))
    filenames = [Path(f.name).name for f in files]
    total_mb = sum(len(c.encode("utf-8")) for c in contents) / 1_000_000
    st.success(f"✅ Read {n_files} files — {total_mb:.2f} MB total")

    if strip_unicode:
        try:
            import unicodedata
            contents = [unicodedata.normalize("NFC", c.replace('\r\n', '\n')) for c in contents]
        except Exception as e:
            log_error(LOG_PATH, "Unicode normalization", e)

    source_id = SCRIPT_MAP.get(source_script, source_script)
    if mode == "Single Target":
        targets = [(target_script, SCRIPT_MAP.get(target_script, target_script))]
    else:
        targets = list(SCRIPT_MAP.items())

    progress = st.progress(0)
    status = st.empty()
    max_chunk_bytes = chunk_mb * 1_000_000

    async def run_all_safe():
        client = None
        if not LOCAL_BINDING:
            if httpx is None:
                raise RuntimeError("httpx required for API mode")
            client = httpx.AsyncClient(limits=httpx.Limits(max_connections=500, max_keepalive_connections=50), timeout=300.0)
        api_sem = asyncio.Semaphore(api_concurrency)

        try:
            total_targets = len(targets)
            for t_idx, (script_name, target_id) in enumerate(targets):
                status.text(f"Converting to {script_name} ({t_idx+1}/{total_targets})")
                safe_name = safe_folder(script_name.replace(" ", "_"))
                folder = OUT_DIR / safe_name
                # create folder safely
                await async_safe_exec(f"mkdir {folder}", asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True), LOG_PATH)

                num_batches = math.ceil(n_files / batch_size)
                for b_idx in range(num_batches):
                    start = b_idx * batch_size
                    end = min(start + batch_size, n_files)
                    batch_contents = contents[start:end]
                    batch_files = filenames[start:end]

                    # create tasks and run them safely
                    tasks = []
                    for i, txt in enumerate(batch_contents):
                        task_name = f"convert {safe_name}/{batch_files[i]}"
                        coro = convert_text_async(source_id, target_id, txt, client, LOCAL_BINDING, max_chunk_bytes, api_semaphore=api_sem, options=local_options)
                        tasks.append(async_safe_exec(task_name, coro, LOG_PATH))

                    results = await asyncio.gather(*tasks)

                    # write results safely
                    write_tasks = []
                    for fname, out_text in zip(batch_files, results):
                        if out_text is None:
                            # log missing result and skip
                            log_error(LOG_PATH, f"Conversion returned None for {fname} -> {script_name}", Exception("result None"))
                            continue
                        dest = folder / fname
                        wt = asyncio.to_thread(write_file_safe, dest, out_text, LOG_PATH)
                        write_tasks.append(async_safe_exec(f"write {dest}", wt, LOG_PATH))

                    # await writes
                    if write_tasks:
                        await asyncio.gather(*write_tasks)

                    progress.progress(((t_idx + (b_idx + 1) / num_batches) / total_targets))
                    gc.collect()

            if client is not None:
                await client.aclose()
            return True
        except Exception as e:
            log_error(LOG_PATH, "run_all_safe top-level", e)
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    pass
            return False

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ok = False
    try:
        ok = loop.run_until_complete(run_all_safe())
    except Exception as e:
        log_error(LOG_PATH, "event_loop_run_until_complete", e)
        ok = False
    finally:
        try:
            loop.close()
        except Exception:
            pass

    # Create ZIP (only files actually on disk)
    try:
        zip_path = OUT_DIR / "transliterated_ultra_fast.zip"
        compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
            for root, dirs, files_in_dir in os.walk(OUT_DIR):
                for f in files_in_dir:
                    full = Path(root) / f
                    if full == zip_path:
                        continue
                    try:
                        arcname = str(full.relative_to(OUT_DIR))
                        # only add files that are normal files
                        if full.is_file():
                            zf.write(full, arcname)
                    except Exception as e:
                        log_error(LOG_PATH, f"zip write {full}", e)
        st.success("✨ Conversion (safe) complete!")
        if zip_path.exists():
            st.write(f"Output ZIP: {zip_path.stat().st_size / 1_000_000:.2f} MB (temp dir: {OUT_DIR})")
            with open(zip_path, "rb") as fh:
                st.download_button("⬇️ Download ZIP", fh, file_name=zip_path.name, mime="application/zip")
        else:
            st.warning("No ZIP created — check errors.log for details.")
    except Exception as e:
        log_error(LOG_PATH, "create_zip", e)
        st.error("Error creating ZIP. Check errors.log.")

    st.info(f"Errors log: {LOG_PATH}")
    st.info("Temporary outputs preserved in case you want to inspect partial results.")

# ----------------- CLI mode (safe) -----------------
if __name__ == "__main__" and "streamlit" not in sys.argv[0]:
    # Usage: python hyper_aksha_opt_safe.py cli /input_folder /output_folder Devanagari Tamil
    if len(sys.argv) >= 2 and sys.argv[1] == "cli":
        if len(sys.argv) < 6:
            print("Usage: python hyper_aksha_opt_safe.py cli <input_folder> <output_folder> <source> <target>")
            sys.exit(1)
        _, _, in_folder, out_folder, source_arg, target_arg = sys.argv[:6]
        in_folder = Path(in_folder)
        out_folder = Path(out_folder)
        LOG_PATH = out_folder / "errors.log"
        out_folder.mkdir(parents=True, exist_ok=True)
        txt_files = list(in_folder.rglob("*.txt"))
        print(f"Found {len(txt_files)} files. Starting conversion (local binding: {LOCAL_BINDING})")

        async def cli_run_safe():
            client = None
            if not LOCAL_BINDING:
                if httpx is None:
                    raise RuntimeError("httpx required for API mode")
                client = httpx.AsyncClient(limits=httpx.Limits(max_connections=200), timeout=300.0)
            try:
                for p in txt_files:
                    try:
                        t = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception as e:
                        log_error(LOG_PATH, f"read {p}", e)
                        continue
                    out = await convert_text_async(source_arg, target_arg, t, client, LOCAL_BINDING, DEFAULT_CHUNK_MB * 1_000_000)
                    if out is None:
                        log_error(LOG_PATH, f"convert {p}", Exception("Conversion returned None"))
                        continue
                    dest = out_folder / p.name
                    ok = sync_safe_exec(f"write {dest}", write_file_safe, LOG_PATH, dest, out, LOG_PATH)
                    if ok:
                        print(f"Wrote {dest}")
                    else:
                        print(f"Failed to write {dest} (see log)")
                if client is not None:
                    await client.aclose()
            except Exception as e:
                log_error(LOG_PATH, "cli_run_safe", e)
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(cli_run_safe())
        finally:
            try:
                loop.close()
            except Exception:
                pass

        print("Done. Check errors.log for any issues.")
