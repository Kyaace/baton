"""Kiwix .zim reader and Wikipedia link sanitizer."""

import re
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    import libzim
except ImportError:
    libzim = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


# Patterns for links that should be deterministically excluded prior to LLM evaluation
EXCLUDED_NAMESPACES = (
    "file:",
    "category:",
    "portal:",
    "help:",
    "wikipedia:",
    "template:",
    "special:",
    "talk:",
    "draft:",
    "user:",
)

# Regex to detect pure years / calendar dates (e.g., "1994", "July 8", "1980s", "20th century")
DATE_PATTERN = re.compile(
    r"^(?:\d{1,4}|\d{1,2}\s+[A-Za-z]+|[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?|\d{2,4}s|\d{1,2}(?:st|nd|rd|th)\s+century)$",
    re.IGNORECASE,
)


def sanitize_article_title(raw_href: str) -> Optional[str]:
    """
    Clean raw href from Kiwix HTML to canonical Wikipedia title.
    Returns None if the link is an anchor, namespace, date, external URL, or invalid.
    """
    if not raw_href:
        return None

    clean = raw_href.strip()
    
    # Filter external links
    if clean.startswith(("http://", "https://", "//", "mailto:", "javascript:")):
        return None

    # Strip relative path markers
    if clean.startswith("../A/"):
        clean = clean[5:]
    elif clean.startswith("./"):
        clean = clean[2:]
    elif clean.startswith("/A/"):
        clean = clean[3:]

    # Strip anchors (#...) and query parameters (?...)
    clean = clean.split("#")[0].split("?")[0]
    if not clean:
        return None

    # Replace URL-encoded spaces / underscores
    title = clean.replace("_", " ").strip()

    # Check excluded namespaces
    lower_title = title.lower()
    if any(lower_title.startswith(ns) for ns in EXCLUDED_NAMESPACES):
        return None

    # Check calendar dates/years
    if DATE_PATTERN.match(title):
        return None

    return title if title else None


class ZimReader:
    """Interface to read articles and extract links from a Kiwix ZIM archive."""

    def __init__(self, zim_path: Path):
        self.zim_path = zim_path
        self.archive = None
        if libzim is not None and zim_path.exists():
            try:
                self.archive = libzim.Archive(str(zim_path))
            except Exception as e:
                print(f"Warning: Failed to load ZIM archive at {zim_path}: {e}")

    @property
    def is_available(self) -> bool:
        """True if libzim is installed and the archive file is loaded."""
        return self.archive is not None

    def get_info(self) -> Dict[str, Any]:
        """Return high-level statistics about the loaded ZIM archive."""
        if not self.is_available:
            return {
                "available": False,
                "path": str(self.zim_path),
                "error": "Archive not loaded or libzim unavailable",
            }
        return {
            "available": True,
            "filename": self.archive.filename,
            "filesize_mb": round(self.archive.filesize / (1024 * 1024), 2),
            "entry_count": self.archive.entry_count,
            "article_count": self.archive.article_count,
            "media_count": getattr(self.archive, "media_count", 0),
            "has_title_index": self.archive.has_title_index,
        }

    def has_article(self, title: str) -> bool:
        """Check if an article exists in the ZIM archive (as direct entry or redirect)."""
        if not self.is_available:
            return False

        key = title.replace(" ", "_")
        for candidate in (f"A/{key}", key, title):
            try:
                if self.archive.has_entry_by_path(candidate):
                    return True
            except Exception:
                pass
        try:
            if hasattr(self.archive, "has_entry_by_title") and self.archive.has_entry_by_title(title):
                return True
        except Exception:
            pass
        return False

    def get_entry(self, title: str) -> Optional[Any]:
        """Retrieve Entry object, following redirects if necessary."""
        if not self.is_available:
            return None

        key = title.replace(" ", "_")
        for candidate in (f"A/{key}", key, title):
            try:
                entry = self.archive.get_entry_by_path(candidate)
                if entry.is_redirect:
                    return entry.get_redirect_entry()
                return entry
            except Exception:
                pass
        return None

    def get_article_html(self, title: str) -> Optional[str]:
        """Fetch raw HTML content for an article title."""
        entry = self.get_entry(title)
        if not entry:
            return None
        try:
            item = entry.get_item()
            return bytes(item.content).decode("utf-8", errors="replace")
        except Exception:
            return None

    def get_random_entry(self) -> Optional[Dict[str, str]]:
        """Fetch a random entry title from the archive."""
        if not self.is_available or not hasattr(self.archive, "get_random_entry"):
            return None
        try:
            entry = self.archive.get_random_entry()
            if entry.is_redirect:
                entry = entry.get_redirect_entry()
            return {"title": entry.title, "path": entry.path}
        except Exception:
            return None

    def extract_article_content(self, title: str, verify_candidates_exist: bool = True) -> Dict[str, Any]:
        """
        Extract summary lead, infobox (if present), and candidate outgoing links.
        """
        html = self.get_article_html(title)
        if not html:
            return {
                "title": title,
                "summary": f"Article '{title}' not found in archive.",
                "infobox": "",
                "candidates": [],
            }

        # Strip scripts and styles from HTML before extracting infobox and paragraphs
        clean_html = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        clean_html = re.sub(r"<style.*?</style>", "", clean_html, flags=re.DOTALL | re.IGNORECASE)

        # 1. Extract infobox text if present
        infobox_text = ""
        infobox_match = re.search(
            r'<table[^>]*class=["\'][^"\']*infobox[^"\']*["\'][^>]*>(.*?)</table>',
            clean_html,
            re.DOTALL | re.IGNORECASE,
        )
        if infobox_match:
            raw_ib = re.sub(r"<[^>]+>", " ", infobox_match.group(1))
            infobox_text = " ".join(raw_ib.split())

        # 2. Extract lead paragraphs
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", clean_html, re.DOTALL | re.IGNORECASE)
        
        summary_paragraphs = []
        for p in paragraphs:
            cleaned_p = re.sub(r"<[^>]+>", "", p)
            cleaned_p = re.sub(r"\[\d+\]", "", cleaned_p)  # Strip citations e.g. [1]
            cleaned_p = " ".join(cleaned_p.split()).strip()
            if len(cleaned_p) > 50:
                summary_paragraphs.append(cleaned_p)
            if len(summary_paragraphs) >= 3:
                break

        summary = "\n\n".join(summary_paragraphs)

        # 3. Extract links
        raw_links = re.findall(r'<a\s+[^>]*href=["\'][^"\']+["\'][^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)
        # Match href attribute specifically
        raw_link_pairs = re.findall(
            r'<a\s+(?:[^>]*?\s+)?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            html,
            re.DOTALL | re.IGNORECASE,
        )

        candidates = []
        seen = set()
        for href, text in raw_link_pairs:
            clean_title = sanitize_article_title(href)
            if not clean_title or clean_title.lower() == title.lower() or clean_title in seen:
                continue

            # Optional check: does target exist in ZIM?
            if verify_candidates_exist and not self.has_article(clean_title):
                continue

            seen.add(clean_title)
            clean_anchor = re.sub(r"<[^>]+>", "", text).strip()
            clean_anchor = " ".join(clean_anchor.split())
            candidates.append({
                "title": clean_title,
                "anchor_text": clean_anchor or clean_title,
            })

        return {
            "title": title,
            "summary": summary,
            "infobox": infobox_text,
            "candidates": candidates,
        }

    def is_connected(self, source: str, target: str) -> Tuple[bool, Optional[str]]:
        """
        Check if source article contains a direct hyperlink to target article.
        Resolves redirects if necessary.
        Returns (is_connected, anchor_text).
        """
        if not self.is_available:
            return False, None

        content = self.extract_article_content(source, verify_candidates_exist=False)
        target_norm = target.strip().lower()

        # Check direct candidate match
        for candidate in content.get("candidates", []):
            cand_title = candidate["title"].strip()
            if cand_title.lower() == target_norm:
                return True, candidate.get("anchor_text", cand_title)

        # Check canonical redirect match if direct match didn't hit
        target_entry = self.get_entry(target)
        target_canonical = target_entry.title.lower() if target_entry else target_norm

        for candidate in content.get("candidates", []):
            cand_title = candidate["title"].strip()
            cand_entry = self.get_entry(cand_title)
            cand_canonical = cand_entry.title.lower() if cand_entry else cand_title.lower()
            if cand_canonical == target_canonical:
                return True, candidate.get("anchor_text", cand_title)

        return False, None

    def is_person(self, title: str) -> bool:
        """
        Determine if an article is about an individual human person using Wikipedia's
        built-in semantic HTML microformats (hCard/vCard, infobox biography, Born/Died headers).
        """
        html = self.get_article_html(title)
        if not html:
            return False

        # 1. Infobox table with "biography vcard" microformat class
        is_bio_vcard = bool(re.search(
            r'<table[^>]*class=["\'][^"\']*\b(biography\s+vcard|vcard\s+biography|infobox\s+person)\b',
            html,
            re.IGNORECASE,
        ))

        # 2. Explicit "Born" header in infobox table row
        has_born_th = bool(re.search(r'<th[^>]*>\s*Born\s*</th>', html, re.IGNORECASE))

        # 3. Explicit "Died" header in infobox table row
        has_died_th = bool(re.search(r'<th[^>]*>\s*Died\s*</th>', html, re.IGNORECASE))

        return is_bio_vcard or has_born_th or has_died_th

    def validate_path(self, path: List[str], mode: str = "hard") -> Dict[str, Any]:
        """
        Validate an ordered chain of articles against ground truth ZIM hyperlinks
        and game mode entity constraints (e.g. Hard Mode person verification).
        """
        if not path or len(path) < 2:
            return {
                "valid": False,
                "hops": 0,
                "error": "Path must contain at least 2 articles (start and target).",
                "steps": [],
            }

        steps = []
        all_connected = True
        all_mode_compliant = True

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            connected, anchor = self.is_connected(src, dst)
            
            # Check personhood of destination (and start on step 0)
            dst_is_person = self.is_person(dst)
            src_is_person = self.is_person(src) if i == 0 else steps[i - 1]["dst_is_person"]

            mode_ok = True
            if mode in ["hard", "medium"]:
                if not dst_is_person or not src_is_person:
                    mode_ok = False
                    all_mode_compliant = False

            steps.append({
                "hop": i + 1,
                "source": src,
                "target": dst,
                "connected": connected,
                "anchor": anchor,
                "src_is_person": src_is_person,
                "dst_is_person": dst_is_person,
                "mode_ok": mode_ok,
            })

            if not connected:
                all_connected = False

        return {
            "valid": all_connected and all_mode_compliant,
            "connected_valid": all_connected,
            "mode_compliant": all_mode_compliant,
            "mode": mode,
            "hops": len(path) - 1,
            "steps": steps,
        }

    # -------------------------------------------------------------------------
    # Research & Full-Text Search Capabilities
    # -------------------------------------------------------------------------

    def search(self, query: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Execute ranked full-text search across the ZIM archive using native Xapian indexing.
        Returns matching entries: [{'title': str, 'path': str, 'snippet': str}, ...]
        """
        if not self.is_available:
            return []

        results = []
        if libzim is not None and hasattr(self.archive, "has_fulltext_index") and self.archive.has_fulltext_index:
            try:
                searcher = libzim.Searcher(self.archive)
                q = libzim.Query().set_query(query)
                search_res = searcher.search(q)
                paths = list(search_res.getResults(offset, limit))
                for p in paths:
                    title = p.replace("_", " ")
                    if title.startswith("A/"):
                        title = title[2:]
                    snippet = ""
                    try:
                        lead = self.read_lead(title, format="text")
                        if lead:
                            snippet = lead.split("\n\n")[0][:180] + ("..." if len(lead) > 180 else "")
                    except Exception:
                        pass
                    results.append({"title": title, "path": p, "snippet": snippet})
                return results
            except Exception:
                pass

        # Fallback: Suggestion / prefix search if fulltext index unavailable
        suggestions = self.suggest_titles(query, limit=limit)
        for s in suggestions:
            results.append({"title": s, "path": s.replace(" ", "_"), "snippet": ""})
        return results

    def suggest_titles(self, prefix: str, limit: int = 10) -> List[str]:
        """
        Return title autocomplete and search suggestions for a given prefix or query.
        """
        if not self.is_available:
            return []

        if libzim is not None and hasattr(libzim, "SuggestionSearcher"):
            try:
                suggester = libzim.SuggestionSearcher(self.archive)
                raw_suggestions = list(suggester.suggest(prefix).getResults(0, limit))
                cleaned = []
                for s in raw_suggestions:
                    title = s.replace("_", " ").strip()
                    if title.startswith("A/"):
                        title = title[2:]
                    if title and title not in cleaned:
                        cleaned.append(title)
                if cleaned:
                    return cleaned
            except Exception:
                pass

        # Fallback: check exact title
        if self.has_article(prefix):
            return [prefix]
        return []

    # -------------------------------------------------------------------------
    # Structured Reading & Table of Contents
    # -------------------------------------------------------------------------

    def get_toc(self, title: str) -> List[Dict[str, Any]]:
        """
        Extract Table of Contents (headings hierarchy) for an article.
        Returns: [{'index': int, 'heading': str, 'level': int, 'id': str}, ...]
        """
        html = self.get_article_html(title)
        if not html:
            return []

        if BeautifulSoup is None:
            # Fallback regex extraction if bs4 unavailable
            matches = re.findall(r'<h([2-6])[^>]*?(?:id=["\']([^"\']*)["\'])?[^>]*>(.*?)</h\1>', html, re.DOTALL | re.IGNORECASE)
            toc = []
            for idx, (lvl, hid, htext) in enumerate(matches, 1):
                clean_text = re.sub(r'<[^>]+>', '', htext).strip()
                if clean_text:
                    toc.append({"index": idx, "heading": clean_text, "level": int(lvl), "id": hid or clean_text.replace(" ", "_")})
            return toc

        soup = BeautifulSoup(html, "html.parser")
        toc = []
        heading_tags = soup.find_all(["h2", "h3", "h4", "h5", "h6"])

        idx = 1
        for h in heading_tags:
            for noise in h.find_all(["span", "sup"]):
                noise_classes = noise.get("class", []) or []
                if any("mw-headline-number" in c or "mw-editsection" in c for c in noise_classes):
                    noise.decompose()
            heading_text = h.get_text().strip()
            if not heading_text or heading_text.lower() in ["navigation menu", "contents", "references", "external links"]:
                continue

            level = int(h.name[1])
            elem_id = h.get("id") or (h.find_parent("section").get("id", "") if h.find_parent("section") else "")
            toc.append({
                "index": idx,
                "heading": heading_text,
                "level": level,
                "id": elem_id or heading_text.replace(" ", "_"),
            })
            idx += 1

        return toc

    def read_lead(self, title: str, format: str = "markdown") -> str:
        """
        Extract the complete introductory lead section (everything before the first H2).
        """
        html = self.get_article_html(title)
        if not html:
            return f"Article '{title}' not found in archive."

        if BeautifulSoup is None:
            raw_lead = html.split("<h2")[0]
            clean = re.sub(r"<[^>]+>", " ", raw_lead)
            return " ".join(clean.split())

        soup = BeautifulSoup(html, "html.parser")

        # Strip non-content and noise elements
        for tag in soup.find_all(["script", "style", "sup", "noscript"]):
            tag.decompose()
        for tag in soup.find_all(["table", "div"], class_=lambda c: c and any(k in " ".join(c if isinstance(c, list) else [c]).lower().split() for k in ["infobox", "navbox", "sidebar", "toc"])):
            tag.decompose()
        for tag in soup.find_all(class_="mw-empty-elt"):
            tag.decompose()

        # Check for first Parsoid section if present
        first_section = soup.find("section")
        if first_section and not first_section.find(["h2", "h3"]):
            lead_soup = first_section
        else:
            lead_parts = []
            main_content = soup.find("div", class_="mw-parser-output") or soup.find("main") or soup.body or soup
            for elem in main_content.children:
                if getattr(elem, "name", None) in ["h2"]:
                    break
                if getattr(elem, "name", None) in ["p", "ul", "ol", "blockquote"]:
                    lead_parts.append(str(elem))
            lead_soup = BeautifulSoup("".join(lead_parts), "html.parser")

        if format == "text":
            paragraphs = [p.get_text().strip() for p in lead_soup.find_all("p") if p.get_text().strip()]
            return "\n\n".join(paragraphs)

        return self._html_soup_to_markdown(lead_soup)

    def read_section(self, title: str, section_name: str, format: str = "markdown") -> Optional[str]:
        """
        Extract and return a specific section by heading name or ID.
        """
        html = self.get_article_html(title)
        if not html:
            return None

        if BeautifulSoup is None:
            return None

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "sup", "noscript"]):
            tag.decompose()

        target_norm = section_name.strip().lower()

        # Strategy 1: Check Parsoid <section> tags
        for sec in soup.find_all("section"):
            heading = sec.find(["h2", "h3", "h4", "h5", "h6"])
            if heading:
                h_text = heading.get_text().strip()
                h_id = heading.get("id", "")
                if target_norm == h_text.lower() or target_norm == h_id.lower() or target_norm in h_text.lower():
                    heading.decompose()
                    return self._html_soup_to_markdown(sec) if format == "markdown" else sec.get_text().strip()

        # Strategy 2: Sibling iteration for non-Parsoid DOM (handles both bare <hN> and <div class="mw-heading">)
        headings = soup.find_all(["h2", "h3", "h4", "h5", "h6"])
        for h in headings:
            h_text = h.get_text().strip()
            h_id = h.get("id", "")
            if target_norm == h_text.lower() or target_norm == h_id.lower() or target_norm in h_text.lower():
                h_level = int(h.name[1])
                start_node = h.parent if h.parent and any("mw-heading" in str(c) for c in (h.parent.get("class") or [])) else h
                collected_nodes = []
                for sib in start_node.next_siblings:
                    if getattr(sib, "name", None):
                        classes = sib.get("class") or []
                        if any("mw-heading" in str(c) for c in classes):
                            sub_h = sib.find(["h2", "h3", "h4", "h5", "h6"])
                            if sub_h and int(sub_h.name[1]) <= h_level:
                                break
                        elif sib.name in [f"h{i}" for i in range(1, h_level + 1)]:
                            break
                    collected_nodes.append(str(sib))
                sec_soup = BeautifulSoup("".join(collected_nodes), "html.parser")
                return self._html_soup_to_markdown(sec_soup) if format == "markdown" else sec_soup.get_text().strip()

        return None

    def read_full(self, title: str, format: str = "markdown", max_chars: Optional[int] = None) -> str:
        """
        Convert and return the entire article in clean Markdown or text.
        """
        html = self.get_article_html(title)
        if not html:
            return f"Article '{title}' not found in archive."

        if BeautifulSoup is None:
            clean = re.sub(r"<[^>]+>", " ", html)
            return " ".join(clean.split())

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "sup", "noscript"]):
            tag.decompose()
        for tag in soup.find_all(["table", "div"], class_=lambda c: c and any(k in " ".join(c if isinstance(c, list) else [c]).lower().split() for k in ["navbox", "sidebar"])):
            tag.decompose()
        for tag in soup.find_all(class_="mw-empty-elt"):
            tag.decompose()

        main_content = soup.find("div", class_="mw-parser-output") or soup.find("main") or soup.body or soup
        res = self._html_soup_to_markdown(main_content) if format == "markdown" else main_content.get_text().strip()
        if max_chars and len(res) > max_chars:
            res = res[:max_chars] + f"\n\n... [Truncated at {max_chars} characters]"
        return res

    # -------------------------------------------------------------------------
    # Structured Knowledge & Taxonomy
    # -------------------------------------------------------------------------

    def get_infobox_dict(self, title: str) -> Dict[str, str]:
        """
        Parse the article's infobox table into structured key-value pairs.
        """
        html = self.get_article_html(title)
        if not html or BeautifulSoup is None:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        infobox = soup.find("table", class_=lambda c: c and "infobox" in str(c))
        if not infobox:
            return {}

        for noise in infobox.find_all(["sup", "style", "script"]):
            noise.decompose()

        kvs = {}
        for tr in infobox.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if th and td:
                key = re.sub(r"\s+", " ", th.get_text()).strip()
                val = re.sub(r"\s+", " ", td.get_text()).strip()
                if key and val:
                    kvs[key] = val
        return kvs

    def get_infobox_markdown(self, title: str) -> str:
        """
        Return the article's infobox formatted as a clean Markdown table.
        """
        kvs = self.get_infobox_dict(title)
        if not kvs:
            return "No infobox found."

        lines = ["| Property | Value |", "| :--- | :--- |"]
        for k, v in kvs.items():
            clean_k = k.replace("|", "/")
            clean_v = v.replace("|", "/")
            lines.append(f"| **{clean_k}** | {clean_v} |")
        return "\n".join(lines)

    def get_categories(self, title: str) -> List[str]:
        """
        Extract Wikipedia categories associated with the article.
        """
        html = self.get_article_html(title)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser") if BeautifulSoup else None
        categories = []

        if soup:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "Category:" in href or "category:" in href:
                    part = re.split(r"Category:", href, flags=re.IGNORECASE)[-1]
                    cat_name = part.replace("_", " ").split("#")[0].split("?")[0].strip()
                    cat_name = urllib.parse.unquote(cat_name).strip()
                    if cat_name and cat_name not in categories:
                        categories.append(cat_name)
        else:
            matches = re.findall(r'href=["\'][^"\']*Category:([^"\']+)["\']', html, re.IGNORECASE)
            for m in matches:
                cat_name = urllib.parse.unquote(m.replace("_", " ")).strip()
                if cat_name and cat_name not in categories:
                    categories.append(cat_name)

        return categories

    def is_disambiguation(self, title: str) -> Tuple[bool, List[str]]:
        """
        Detect whether an article is a disambiguation page and extract candidate options.
        Returns: (is_disambiguation, candidate_options)
        """
        html = self.get_article_html(title)
        if not html:
            return False, []

        is_disambig = False
        if "(disambiguation)" in title.lower():
            is_disambig = True

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            if soup.find(class_=lambda c: c and "disambig" in str(c)) or soup.find(id="disambig"):
                is_disambig = True
            elif re.search(r"\bmay\s+refer\s+to\s*:", html, re.IGNORECASE):
                is_disambig = True

            if not is_disambig:
                return False, []

            choices = []
            for li in soup.find_all("li"):
                a = li.find("a", href=True)
                if a:
                    clean = sanitize_article_title(a["href"])
                    if clean and clean.lower() != title.lower() and clean not in choices:
                        choices.append(clean)
            return True, choices

        # Regex fallback
        if re.search(r'class=["\'][^"\']*disambig[^"\']*["\']', html, re.IGNORECASE) or re.search(r"\bmay\s+refer\s+to\s*:", html, re.IGNORECASE):
            is_disambig = True

        if not is_disambig:
            return False, []

        raw_links = re.findall(r'<a\s+[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)
        choices = []
        for href in raw_links:
            clean = sanitize_article_title(href)
            if clean and clean.lower() != title.lower() and clean not in choices:
                choices.append(clean)
        return True, choices

    def _html_soup_to_markdown(self, soup_node) -> str:
        """
        Convert BeautifulSoup DOM fragment to clean, token-efficient Markdown.
        """
        if not soup_node:
            return ""

        output_lines = []
        for elem in soup_node.children:
            name = getattr(elem, "name", None)
            if not name:
                text = str(elem).strip()
                if text:
                    output_lines.append(text)
                continue

            if name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                lvl = int(name[1])
                prefix = "#" * lvl
                text = elem.get_text().strip()
                if text:
                    output_lines.append(f"\n{prefix} {text}\n")
            elif name == "p":
                text = elem.get_text().strip()
                if text:
                    output_lines.append(f"\n{text}\n")
            elif name in ["ul", "ol"]:
                for idx, li in enumerate(elem.find_all("li", recursive=False), 1):
                    li_text = li.get_text().strip()
                    if li_text:
                        bullet = f"{idx}." if name == "ol" else "-"
                        output_lines.append(f"{bullet} {li_text}")
                output_lines.append("")
            elif name == "blockquote":
                text = elem.get_text().strip()
                if text:
                    output_lines.append(f"> {text}\n")
            elif name == "table":
                rows = elem.find_all("tr")
                if rows:
                    table_lines = []
                    header_done = False
                    for r_idx, tr in enumerate(rows):
                        cells = tr.find_all(["th", "td"])
                        if cells:
                            row_str = " | ".join(re.sub(r"\s+", " ", c.get_text().strip()).replace("|", "/") for c in cells)
                            table_lines.append(f"| {row_str} |")
                            if not header_done:
                                sep = " | ".join("---" for _ in cells)
                                table_lines.append(f"| {sep} |")
                                header_done = True
                    if table_lines:
                        output_lines.append("\n" + "\n".join(table_lines) + "\n")
            else:
                sub_text = self._html_soup_to_markdown(elem)
                if sub_text:
                    output_lines.append(sub_text)

        merged = "\n".join(output_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", merged).strip()
        return cleaned
