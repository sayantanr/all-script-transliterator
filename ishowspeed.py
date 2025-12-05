# -*- coding: utf-8 -*-
"""
HYPER-OPTIMIZED Aksharamukha Transliterator - Full code
Single-file Streamlit app + high-performance transliteration engine.
- Uses local aksharamukha binding (ProcessPoolExecutor) if installed.
- Otherwise uses httpx.AsyncClient with connection pooling (API mode).
- Disk-backed caching, streaming ZIP output, chunking, and parallel file reading.
- Tunable concurrency exposed in UI.
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
import gc
from pathlib import Path
from functools import partial, lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple

# UI
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

# ----------------- FULL SCRIPT MAP (paste of user's original map) -----------------
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
CACHE_DIR = TEMP_BASE / f"aksha_cache_{os.getpid()}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default concurrency
DEFAULT_IO_WORKERS = max(4, (os.cpu_count() or 4))
DEFAULT_BATCH_SIZE = 50
DEFAULT_CHUNK_MB = 4
DEFAULT_API_CONCURRENCY = 12

# Process pool for local binding
LOCAL_POOL = None
if LOCAL_BINDING:
    # Use half of CPU cores for pool to avoid OS starvation
    pool_workers = max(1, (os.cpu_count() or 2) // 2)
    LOCAL_POOL = ProcessPoolExecutor(max_workers=pool_workers)

# ----------------- Utilities -----------------
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

def read_file_fast(f):
    """Fast safe read of uploaded file-like object"""
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
    """Split a string into utf-8 byte-bounded chunks"""
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

# ----------------- Backends -----------------
def _local_transform(src: str, tgt: str, text: str, options=None) -> str:
    """Synchronous local aksharamukha call. Safe single-chunk call for ProcessPoolExecutor."""
    try:
        if options:
            return ak_trans.process(src, tgt, text, **options)
        else:
            return ak_trans.process(src, tgt, text)
    except Exception:
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

# ----------------- High-level convert per-file -----------------
async def convert_text_async(source_id: str, target_id: str, text: str, client: "httpx.AsyncClient", use_local: bool, max_chunk_bytes: int, api_semaphore: asyncio.Semaphore=None, options=None):
    """
    Convert a single file text to target.
    - If use_local: use LOCAL_POOL to run _local_transform on chunks concurrently.
    - Else: use httpx.AsyncClient and run requests concurrently on chunks (bounded by api_semaphore).
    """
    # quick cache key uses preview of text (first 4KB)
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
        # API mode
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

# ----------------- Streamlit UI & Orchestration -----------------
st.set_page_config(page_title="⚡ HYPER-OPTIMIZED Aksharamukha Transliterator", layout="wide")
st.title("⚡ HYPER-OPTIMIZED Aksharamukha Transliterator — Full Engine")

if LOCAL_BINDING:
    st.success("🚀 Local binding: using aksharamukha package (ProcessPoolExecutor)")
else:
    st.info("📡 API mode (httpx). Install `aksharamukha` for local speed: pip install aksharamukha")

# Upload
uploaded_files = st.file_uploader("📤 Upload .txt files (you can upload many)", type="txt", accept_multiple_files=True)

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
strip_unicode = st.checkbox("🧹 Normalize text (NFC) before transliteration", True)
disable_spellcheck = st.checkbox("⚡ Disable extra processing options (recommended)", True)

# Options to pass to local binding if used
local_options = {}
if disable_spellcheck:
    # These option names may vary by aksharamukha version; adapt if needed.
    local_options = {"pre_options": ["SanskritMode"], "post_options": ["DoNotReplacePunctuation"]}

# Action button
if st.button("⚡ START HYPER-FAST CONVERSION"):
    if not uploaded_files:
        st.error("❌ Please upload at least one file")
        st.stop()

    files = uploaded_files[:10000]
    n_files = len(files)
    st.info(f"Reading {n_files} files using {workers} threads...")

    # Fast parallel read
    with ThreadPoolExecutor(max_workers=workers) as tpe:
        contents = list(tpe.map(read_file_fast, files))
    filenames = [Path(f.name).name for f in files]
    total_mb = sum(len(c.encode("utf-8")) for c in contents) / 1_000_000
    st.success(f"✅ Read {n_files} files — {total_mb:.2f} MB total")

    # Optional normalization
    if strip_unicode:
        contents = [c.replace('\r\n', '\n') for c in contents]  # small normalization
        # use NFC normalization if available
        try:
            import unicodedata
            contents = [unicodedata.normalize("NFC", c) for c in contents]
        except Exception:
            pass

    # Prepare targets
    source_id = SCRIPT_MAP.get(source_script, source_script)
    if mode == "Single Target":
        targets = [(target_script, SCRIPT_MAP.get(target_script, target_script))]
    else:
        targets = list(SCRIPT_MAP.items())

    # Temp output dir (disk-backed)
    OUT_DIR = TEMP_BASE / f"aksha_out_{os.getpid()}_{int(time.time())}"
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0)
    status = st.empty()
    max_chunk_bytes = chunk_mb * 1_000_000

    async def run_all():
        # create httpx async client if API mode
        client = None
        if not LOCAL_BINDING:
            if httpx is None:
                raise RuntimeError("httpx is required for API mode. pip install httpx")
            client = httpx.AsyncClient(limits=httpx.Limits(max_connections=500, max_keepalive_connections=50), timeout=300.0)
        api_sem = asyncio.Semaphore(api_concurrency)

        try:
            total_targets = len(targets)
            for t_idx, (script_name, target_id) in enumerate(targets):
                status.text(f"Converting to {script_name} ({t_idx+1}/{total_targets})")
                folder = OUT_DIR / script_name.replace(" ", "_").replace("(", "").replace(")", "")
                folder.mkdir(parents=True, exist_ok=True)

                num_batches = math.ceil(n_files / batch_size)
                for b_idx in range(num_batches):
                    start = b_idx * batch_size
                    end = min(start + batch_size, n_files)
                    batch_contents = contents[start:end]
                    batch_files = filenames[start:end]

                    # create async tasks for this batch
                    tasks = [
                        convert_text_async(
                            source_id, target_id, txt, client, LOCAL_BINDING, max_chunk_bytes, api_semaphore=api_sem, options=local_options
                        ) for txt in batch_contents
                    ]
                    # run concurrently
                    results = await asyncio.gather(*tasks)

                    # write results to disk
                    for fname, out_text in zip(batch_files, results):
                        dest = folder / fname
                        try:
                            dest.write_text(out_text, encoding="utf-8")
                        except Exception:
                            dest.write_text(out_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"), encoding="utf-8")

                    # progress update
                    progress.progress(((t_idx + (b_idx + 1) / num_batches) / total_targets))
                    # small GC trim
                    gc.collect()

            if client is not None:
                await client.aclose()
            return True
        except Exception as e:
            if client is not None:
                await client.aclose()
            raise

    # Run asyncio pipeline
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_all())
    except Exception as e:
        st.error(f"Error during conversion: {e}")
        raise
    finally:
        loop.close()

    # Create ZIP on disk
    zip_path = OUT_DIR / "transliterated_ultra_fast.zip"
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
        for root, dirs, files_in_dir in os.walk(OUT_DIR):
            for f in files_in_dir:
                full = Path(root) / f
                if full == zip_path:
                    continue
                arcname = str(full.relative_to(OUT_DIR))
                zf.write(full, arcname)

    st.success("✨ Conversion complete!")
    st.write(f"Output ZIP: {zip_path.stat().st_size / 1_000_000:.2f} MB (temp dir: {OUT_DIR})")
    with open(zip_path, "rb") as fh:
        st.download_button("⬇️ Download ZIP", fh, file_name=zip_path.name, mime="application/zip")

    st.info("Tip: Keep this terminal open until download completes. Temporary files preserved in case of re-run.")

# ----------------- Optional: CLI mode for batch servers -----------------
if __name__ == "__main__" and "streamlit" not in sys.argv[0]:
    # Simple CLI to transliterate a folder -> output folder (non-UI)
    # Usage: python hyper_aksha_opt.py cli /input_folder /output_folder Devanagari Tamil
    if len(sys.argv) >= 2 and sys.argv[1] == "cli":
        if len(sys.argv) < 6:
            print("Usage: python hyper_aksha_opt.py cli <input_folder> <output_folder> <source> <target>")
            sys.exit(1)
        _, _, in_folder, out_folder, source_arg, target_arg = sys.argv[:6]
        in_folder = Path(in_folder)
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        # read files
        txt_files = list(in_folder.rglob("*.txt"))
        print(f"Found {len(txt_files)} files. Starting conversion (local binding: {LOCAL_BINDING})")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def cli_run():
            client = None
            if not LOCAL_BINDING:
                if httpx is None:
                    raise RuntimeError("httpx required for API mode")
                client = httpx.AsyncClient(limits=httpx.Limits(max_connections=200), timeout=300.0)
            try:
                for p in txt_files:
                    t = p.read_text(encoding="utf-8", errors="ignore")
                    out = await convert_text_async(source_arg, target_arg, t, client, LOCAL_BINDING, DEFAULT_CHUNK_MB * 1_000_000)
                    dest = out_folder / p.name
                    dest.write_text(out, encoding="utf-8")
                    print(f"Wrote {dest}")
                if client is not None:
                    await client.aclose()
            except Exception as e:
                print("Error:", e)
                if client is not None:
                    await client.aclose()
                raise

        loop.run_until_complete(cli_run())
        print("Done.")
