#!/usr/bin/env python3
"""
Unified PDF downloader for entire polymath corpus.
Tries multiple sources: Primary URL → Unpaywall → arXiv → bioRxiv → EuropePMC
"""

import sqlite3
import requests
import time
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

DB_PATH = "/home/user/.claude/mcp/research_lab.db"
PDF_DIR = Path("/home/user/work/polymath_pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path("/home/user/rubric-rewards-training/logs/download_all.log")

# Rate limiting
DELAY_PRIMARY = 2
DELAY_API = 3

# API endpoints
ARXIV_API = "http://export.arxiv.org/api/query"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
UNPAYWALL_EMAIL = "max.r.van.belkum@vanderbilt.edu"


def log(msg):
    """Log to file only (stdout goes to same file via nohup)"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)


def clean_title(title):
    """Clean title for search query"""
    clean = re.sub(r'[^\w\s]', ' ', title)
    return ' '.join(clean.split())[:100]


def download_pdf(url, paper_id):
    """Download PDF and save to disk"""
    if not url or not url.startswith("http"):
        return False, "Invalid URL"
    try:
        resp = requests.get(url, timeout=30, allow_redirects=True, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        content_type = resp.headers.get('Content-Type', '')

        if resp.status_code == 200 and len(resp.content) > 5000:
            is_pdf = resp.content[:4] == b'%PDF' or 'pdf' in content_type.lower()
            if is_pdf:
                pdf_path = PDF_DIR / f"{paper_id}.pdf"
                pdf_path.write_bytes(resp.content)
                return True, len(resp.content)
            return False, f"Not PDF ({content_type[:20]})"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)[:50]


def try_unpaywall(doi):
    """Try Unpaywall API"""
    if not doi:
        return None
    try:
        resp = requests.get(f"{UNPAYWALL_API}/{doi}",
                          params={"email": UNPAYWALL_EMAIL}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("is_oa"):
            best = data.get("best_oa_location", {})
            return best.get("url_for_pdf") or best.get("url")
    except:
        pass
    return None


def try_arxiv(title):
    """Try arXiv API"""
    try:
        query = f'ti:"{clean_title(title)}"'
        resp = requests.get(ARXIV_API, params={
            "search_query": query, "max_results": 2
        }, timeout=10)
        if resp.status_code != 200:
            return None

        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        title_lower = title.lower()[:40]

        for entry in root.findall('atom:entry', ns):
            entry_title = entry.find('atom:title', ns).text.strip().lower()
            if title_lower in entry_title or entry_title[:40] in title_lower:
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        return link.get('href')
    except:
        pass
    return None


def try_europepmc(title):
    """Try Europe PMC API"""
    try:
        resp = requests.get(f"{EUROPEPMC_API}/search", params={
            "query": f'TITLE:"{clean_title(title)[:60]}"',
            "pageSize": 3, "format": "json"
        }, timeout=10)
        if resp.status_code != 200:
            return None

        results = resp.json().get("resultList", {}).get("result", [])
        title_lower = title.lower()[:40]

        for paper in results:
            paper_title = paper.get("title", "").lower()
            if title_lower in paper_title or paper_title[:40] in title_lower:
                pmcid = paper.get("pmcid")
                if pmcid:
                    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    except:
        pass
    return None


def process_paper(paper_id, title, primary_url, metadata):
    """Try all sources for a single paper"""

    # Extract DOI from metadata
    doi = None
    if metadata:
        try:
            doi = json.loads(metadata).get("doi")
        except:
            pass

    # 1. Try primary URL
    if primary_url and primary_url.startswith("http"):
        success, result = download_pdf(primary_url, paper_id)
        if success:
            return "PRIMARY", result
        time.sleep(DELAY_PRIMARY)

    # 2. Try Unpaywall (fast, by DOI)
    if doi:
        url = try_unpaywall(doi)
        if url:
            success, result = download_pdf(url, paper_id)
            if success:
                return "UNPAYWALL", result
        time.sleep(1)

    # 3. Try arXiv (by title)
    time.sleep(DELAY_API)
    url = try_arxiv(title)
    if url:
        success, result = download_pdf(url, paper_id)
        if success:
            return "ARXIV", result

    # 4. Try Europe PMC (by title, for PMC papers)
    time.sleep(DELAY_API)
    url = try_europepmc(title)
    if url:
        success, result = download_pdf(url, paper_id)
        if success:
            return "PMC", result

    return None, "All sources failed"


def main():
    """Main download loop"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get all pending papers
    c.execute("""
        SELECT id, title, pdf_url, metadata FROM paper_queue
        WHERE status='PENDING'
        ORDER BY priority DESC, citation_count DESC
    """)
    papers = c.fetchall()

    total = len(papers)
    log(f"Starting download of {total} papers")
    log(f"PDF directory: {PDF_DIR}")
    log("=" * 60)

    stats = {"PRIMARY": 0, "UNPAYWALL": 0, "ARXIV": 0, "PMC": 0, "FAILED": 0}

    for i, (paper_id, title, pdf_url, metadata) in enumerate(papers):
        # Progress every 50 papers
        if i % 50 == 0:
            log(f"Progress: {i}/{total} | Success: {sum(v for k,v in stats.items() if k != 'FAILED')} | Failed: {stats['FAILED']}")

        source, result = process_paper(paper_id, title, pdf_url, metadata)

        if source:
            c.execute("UPDATE paper_queue SET status='DOWNLOADED', pdf_url=? WHERE id=?",
                     (pdf_url, paper_id))
            conn.commit()
            stats[source] += 1
            if i % 10 == 0:  # Log every 10th success
                log(f"  [{i}] {source}: {title[:50]}... ({result//1024}KB)")
        else:
            c.execute("UPDATE paper_queue SET status='FAILED' WHERE id=?", (paper_id,))
            conn.commit()
            stats["FAILED"] += 1

    conn.close()

    log("=" * 60)
    log("DOWNLOAD COMPLETE")
    log(f"Primary URL: {stats['PRIMARY']}")
    log(f"Unpaywall: {stats['UNPAYWALL']}")
    log(f"arXiv: {stats['ARXIV']}")
    log(f"PMC: {stats['PMC']}")
    log(f"Failed: {stats['FAILED']}")
    log(f"Total PDFs: {sum(v for k,v in stats.items() if k != 'FAILED')}")

    # Final disk usage
    files = list(PDF_DIR.glob('*.pdf'))
    size_gb = sum(f.stat().st_size for f in files) / (1024**3)
    log(f"Disk usage: {len(files)} files, {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
