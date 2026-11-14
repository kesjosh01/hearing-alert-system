"""Collect public, explicitly licensed target audio using existing stdlib/ffmpeg.

No software, packages, browser extensions, credentials or user audio are fetched
or uploaded. Source files and metadata snapshots are append-only local outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/expansion_20261114"
PAGES = DEST / "pages"
REPORT = ROOT / "reports/expansion_20261114"
AGENT = "Mozilla/5.0 (public-audio-research)"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def text(value):
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split())


def get(url, path=None, limit=4_000_000):
    if path is not None and path.is_file():
        return path.read_bytes(), "cached"
    if not url.startswith("https://"):
        raise ValueError("HTTPS public sources only")
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=35) as response:
        data = response.read(limit + 1)
        kind = response.headers.get("Content-Type", "")
    if len(data) > limit:
        raise ValueError("Source exceeds collection size limit")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data, kind


def gongu_detail(sid):
    url = f"https://gongu.copyright.or.kr/gongu/wrt/wrt/view.do?menuNo=200020&wrtSn={sid}"
    raw, _ = get(url, PAGES / f"gongu_{sid}.html")
    page = raw.decode("utf-8", errors="replace")
    fields = {text(k): text(v) for k, v in re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", page, re.S)}
    title = fields.get("저작물명", "")
    if not title:
        raise ValueError("No authoritative work title")
    license_match = re.search(r"https://creativecommons.org/licenses/by/4\.0/deed.ko", page)
    if license_match is None or 'img_license21.png' not in page:
        raise ValueError("Only verified CC BY 4.0 GongU files are accepted")
    media = re.search(r'data-src="([^\"]*wrtFileMediaPlay[^\"]*)"', page)
    if media is None:
        raise ValueError("Public audio player has no media source")
    return {"sample_id": f"gongu-{sid}", "source_dataset": "GongU", "source_recording_id": str(sid),
            "title": title, "author": fields.get("저작(권)자", "unknown"), "source_url": url,
            "media_url": urllib.parse.urljoin(url, html.unescape(media[1])),
            "license_id": "CC-BY-4.0", "license_url": license_match[0],
            "catalog_fields": fields, "origin": "KR_public_catalog", "device_country_verified": False}


def bsb_catalog(term):
    url = "https://bigsoundbank.com/search?" + urllib.parse.urlencode({"q": term})
    page = get(url, PAGES / f"bsb_search_{term.replace(' ', '_')}.html")[0].decode("utf-8", errors="replace")
    links = {}
    for href, label in re.findall(r'<a[^>]*href=[\"\']([^\"\']+)[\"\'][^>]*>(.*?)</a>', page, re.S | re.I):
        if re.search(r"-s\d+\.html", href):
            title = text(label)
            if title and title != "more & Downloads":
                links[urllib.parse.urljoin(url, html.unescape(href))] = title
    return links


def bsb_detail(url):
    sid = re.search(r"-s(\d+)\.html", url)[1]
    page = get(url, PAGES / f"bsb_{sid}.html")[0].decode("utf-8", errors="replace")
    title = text(re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)[1])
    flat = text(page)
    if "CC0 (public domain)" not in flat:
        raise ValueError("Missing item-specific CC0 license")
    media = re.findall(r'(?:src|href)=[\"\']([^\"\']+\.(?:mp3|wav)(?:\?[^\"\']*)?)[\"\']', page, re.I)
    media = [urllib.parse.urljoin(url, html.unescape(u)) for u in media]
    if not media:
        # Public player URL may be in its data-src attribute.
        media = [urllib.parse.urljoin(url, html.unescape(u)) for u in re.findall(r'data-src=[\"\']([^\"\']+)[\"\']', page) if '.mp3' in u or '.wav' in u]
    if not media:
        raise ValueError("No advertised public audio URL")
    author = re.search(r"Author:\s*([^<]{1,100})", flat)
    description = re.search(r'<meta[^>]*name=[\"\']description[\"\'][^>]*content=[\"\']([^\"\']+)', page, re.I)
    return {"sample_id": f"bsb-new-{sid}", "source_dataset": "BigSoundBank", "source_recording_id": sid,
            "title": title, "author": author[1].split("Sound number:")[0].strip() if author else "See item page",
            "source_url": url, "media_url": media[0], "license_id": "CC0-1.0",
            "license_url": "https://bigsoundbank.com/licenses.html", "description": html.unescape(description[1]) if description else "",
            "origin": "overseas_public_catalog", "device_country_verified": False}


def download(record):
    raw_dir = DEST / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext = ".mp3" if ".mp3" in record["media_url"] else ".audio"
    path = raw_dir / (record["sample_id"] + ext)
    data, content_type = get(record["media_url"], path, 60_000_000)
    if data[:1] == b"<" or len(data) < 200:
        raise ValueError("Media endpoint returned HTML or empty content")
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,format_name:stream=codec_type,sample_rate,channels", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)
    duration = float(probe["format"]["duration"])
    if not 0.1 <= duration <= 180 or not any(s.get("codec_type") == "audio" for s in probe["streams"]):
        raise ValueError("Audio duration/stream outside collection scope")
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    record.update(audio_root="dscnn", relative_path=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(data).hexdigest(),
                  decoded_pcm_sha256=hashlib.sha256(pcm).hexdigest(), duration_s=duration, bytes=len(data),
                  content_type=content_type, probe=probe, label_basis="explicit_public_catalog_target_description",
                  human_listening_verified=False, redistribution_requires_attribution=record["license_id"] == "CC-BY-4.0")
    return record


def run(mode):
    PAGES.mkdir(parents=True, exist_ok=True)
    if mode == "scout-foreign":
        found = {}
        for term in ("fire alarm", "smoke detector", "doorbell"):
            found.update(bsb_catalog(term))
        save_json(PAGES / "bsb_candidates.json", found)
        for url, title in found.items():
            if any(w in title.lower() for w in ("smoke", "fire alarm", "evacuation", "doorbell", "intercom")) and 'old doorbell' not in title.lower():
                print(url, title, flush=True)
        return
    if mode == "foreign":
        candidates = json.loads((PAGES / "bsb_candidates.json").read_text())
        fire_ids = {"2249", "3411", "3413", "0800", "0268", "3412", "0269", "1151", "2250"}
        bell_ids = {"2365", "3588", "3046", "1605", "0161", "0160"}
        records, failures = [], []
        for url in candidates:
            sid = re.search(r"-s(\d+)\.html", url)[1]
            if sid not in fire_ids | bell_ids:
                continue
            try:
                item = bsb_detail(url)
                label = "fire_alarm" if sid in fire_ids else "doorbell"
                if label == "fire_alarm" and not any(w in item["title"].lower() for w in ("smoke detector", "fire alarm", "evacuation alarm")):
                    raise ValueError("Ambiguous fire-alarm category")
                if sid in {"0800", "1151"}:
                    group = "bsb-smoke-detector-family"
                elif sid in {"0268", "0269", "2249", "2250"}:
                    group = "bsb-community-fire-alarm-family"
                elif sid in {"3411", "3412", "3413"}:
                    group = "bsb-concert-hall-evacuation-family"
                elif sid in {"0160", "0161"}:
                    group = "bsb-electronic-house-doorbell-family"
                else:
                    group = f"bsb-source-{int(sid)}"
                item.update(label=label, label_id=0 if label == "fire_alarm" else 1, target_domain="general",
                            sampler_pool="auxiliary_target", split_group_id=group,
                            source_group_basis="conservative_device_series; related recordings share group")
                records.append(download(item))
                save_json(DEST / "foreign_downloads.json", records)
                print("FOREIGN_AUDIO_SAVED", sid, item["title"], item["bytes"], flush=True)
            except Exception as error:
                failures.append({"id": sid, "url": url, "error": f"{type(error).__name__}: {error}"})
                save_json(DEST / "foreign_failures.json", failures)
                print("FOREIGN_SOURCE_UNAVAILABLE", sid, str(error)[:180], flush=True)
        save_json(DEST / "foreign_downloads.json", records)
        save_json(DEST / "foreign_failures.json", failures)
        print("FOREIGN_COLLECTION_COMPLETE", len(records), "failures", len(failures), flush=True)
        return
    if mode == "gongu":
        candidates = json.loads((PAGES / "gongu_candidates.json").read_text())
        # Balance target clips after siren relabeling; avoid flooding the dataset
        # with all 21 closely related intercom variations from one creator.
        picked = [(sid, title) for sid, title in candidates.items() if re.fullmatch(r"초인종\s*[A-J]", title)]
        picked += [(sid, title) for sid, title in candidates.items() if re.fullmatch(r"인터폰벨\s*(?:[1-9]|10|11|12)", title)]
        picked += [(sid, title) for sid, title in candidates.items() if sid in ("13254557", "13220897")]
        records, failures = [], []
        for sid, title in picked:
            try:
                item = gongu_detail(sid)
                item.update(label="doorbell", label_id=1, target_domain="ko_public_catalog", sampler_pool="domestic_target",
                            split_group_id="gongu-kim-doorbell-series" if re.fullmatch(r"초인종\s*[A-J]", title) else "gongu-kim-intercom-series" if '인터폰' in title else f"gongu-source-{sid}",
                            source_group_basis="conservative_creator_and_device_series; source independence not claimed")
                records.append(download(item))
                save_json(DEST / "gongu_downloads.json", records)
                print("GONGU_AUDIO_SAVED", sid, item["title"], item["bytes"], flush=True)
            except Exception as error:
                failures.append({"id": sid, "title": title, "error": f"{type(error).__name__}: {error}"})
                save_json(DEST / "gongu_failures.json", failures)
                print("GONGU_SOURCE_UNAVAILABLE", sid, str(error)[:180], flush=True)
        save_json(DEST / "gongu_downloads.json", records)
        save_json(DEST / "gongu_failures.json", failures)
        print("GONGU_COLLECTION_COMPLETE", len(records), "failures", len(failures), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gongu", "scout-foreign", "foreign"))
    run(parser.parse_args().mode)
