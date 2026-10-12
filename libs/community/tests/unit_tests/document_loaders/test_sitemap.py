"""Unit tests for the sitemap loader (no network access)."""

import importlib.util
import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from langchain_community.document_loaders.sitemap import SitemapLoader

pytest.importorskip("bs4")

pytestmark = pytest.mark.requires("bs4")

# `lxml` is an optional extra that is not part of the core unit-test
# environment. The tests below never fetch or parse with the loader's own
# "xml" parser (fetching is stubbed out), so fall back to the stdlib-backed
# "html.parser" when lxml is unavailable and stub out the constructor's
# `import lxml` availability check.
_HAS_LXML = importlib.util.find_spec("lxml") is not None
_PARSER = "xml" if _HAS_LXML else "html.parser"

SITEMAP_INDEX_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>{loc}</loc>
  </sitemap>
</sitemapindex>
"""

CHILD_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/page-1</loc>
  </url>
</urlset>
"""


@pytest.fixture(autouse=True)
def _lxml_available(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _HAS_LXML:
        monkeypatch.setitem(sys.modules, "lxml", types.ModuleType("lxml"))


def _soup(content: str) -> Any:
    import bs4

    return bs4.BeautifulSoup(content, _PARSER)


class _RecordingSitemapLoader(SitemapLoader):
    """Sitemap loader that records `scrape_all` calls instead of fetching."""

    def __init__(
        self, *args: Any, responses: Optional[Dict[str, str]] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scraped_urls: List[str] = []
        self.responses: Dict[str, str] = responses or {}

    def scrape_all(self, urls: List[str], parser: Any = None) -> List[Any]:
        self.scraped_urls.extend(urls)
        return [_soup(self.responses.get(url, CHILD_SITEMAP)) for url in urls]

    def _scrape(
        self,
        url: str,
        parser: Any = None,
        bs_kwargs: Optional[dict] = None,
    ) -> Any:
        self.scraped_urls.append(url)
        return _soup(self.responses[url])


def test_parse_sitemap_skips_cross_domain_nested_sitemap() -> None:
    """A nested <sitemap> on another domain must not be fetched.

    `restrict_to_same_domain` is documented as confining the loader to the
    sitemap's own domain. Nested sitemap-index entries are fetched
    recursively, so skipping the check there would allow a server-side
    request to an arbitrary (e.g. internal) host.
    """
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=True,
    )

    els = loader.parse_sitemap(
        _soup(
            SITEMAP_INDEX_TEMPLATE.format(
                loc="http://169.254.169.254/latest/meta-data/"
            )
        )
    )

    assert els == []
    assert loader.scraped_urls == []


@pytest.mark.parametrize(
    "loc",
    [
        "http://127.0.0.1:8080/admin/sitemap.xml",
        "http://localhost/internal-sitemap.xml",
        # Userinfo trick: the real host is the metadata endpoint.
        "https://example.com@169.254.169.254/latest/meta-data/",
        # Look-alike host sharing the sitemap's domain as a prefix.
        "https://example.com.internal.corp/sitemap.xml",
        # Leading/trailing whitespace must not smuggle an internal host past.
        "\n      http://169.254.169.254/latest/meta-data/\n  ",
    ],
)
def test_parse_sitemap_skips_internal_nested_sitemap_variants(loc: str) -> None:
    """Common SSRF host-confusion payloads in a nested <sitemap> are not fetched."""
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=True,
    )

    els = loader.parse_sitemap(_soup(SITEMAP_INDEX_TEMPLATE.format(loc=loc)))

    assert els == []
    assert loader.scraped_urls == []


def test_parse_sitemap_skips_cross_domain_sitemap_at_deeper_level() -> None:
    """A same-domain child sitemap cannot chain into an internal host."""
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=True,
        responses={
            "https://example.com/nested.xml": SITEMAP_INDEX_TEMPLATE.format(
                loc="http://169.254.169.254/latest/meta-data/"
            ),
        },
    )

    els = loader.parse_sitemap(
        _soup(SITEMAP_INDEX_TEMPLATE.format(loc="https://example.com/nested.xml"))
    )

    assert els == []
    assert loader.scraped_urls == ["https://example.com/nested.xml"]


def test_lazy_load_does_not_fetch_cross_domain_nested_sitemap() -> None:
    """End to end: loading a sitemap index never requests an internal host."""
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=True,
        responses={
            "https://example.com/sitemap.xml": SITEMAP_INDEX_TEMPLATE.format(
                loc="http://169.254.169.254/latest/meta-data/"
            ),
        },
    )

    docs = list(loader.lazy_load())

    assert docs == []
    assert loader.scraped_urls == ["https://example.com/sitemap.xml"]


def test_parse_sitemap_follows_same_domain_nested_sitemap() -> None:
    """A nested <sitemap> on the same domain is still fetched and parsed."""
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=True,
    )

    els = loader.parse_sitemap(
        _soup(SITEMAP_INDEX_TEMPLATE.format(loc="https://example.com/nested.xml"))
    )

    assert loader.scraped_urls == ["https://example.com/nested.xml"]
    assert [el["loc"] for el in els] == ["https://example.com/page-1"]


def test_parse_sitemap_allows_cross_domain_nested_sitemap_when_unrestricted() -> None:
    """With `restrict_to_same_domain=False` the previous behavior is preserved."""
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=False,
    )

    els = loader.parse_sitemap(
        _soup(SITEMAP_INDEX_TEMPLATE.format(loc="https://other.example.org/nested.xml"))
    )

    assert loader.scraped_urls == ["https://other.example.org/nested.xml"]
    assert [el["loc"] for el in els] == ["https://example.com/page-1"]


def test_parse_sitemap_strips_whitespace_from_nested_location() -> None:
    """Whitespace around a nested <loc> must not defeat the domain check."""
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=True,
    )

    els = loader.parse_sitemap(
        _soup(
            SITEMAP_INDEX_TEMPLATE.format(
                loc="\n      https://example.com/nested.xml\n  "
            )
        )
    )

    assert loader.scraped_urls == ["https://example.com/nested.xml"]
    assert [el["loc"] for el in els] == ["https://example.com/page-1"]


def test_parse_sitemap_local_sitemap_is_not_domain_restricted() -> None:
    """Local sitemaps keep their documented (unrestricted) behavior."""
    loader = _RecordingSitemapLoader(
        "https://example.com/sitemap.xml",
        restrict_to_same_domain=True,
        is_local=True,
    )

    els = loader.parse_sitemap(
        _soup(SITEMAP_INDEX_TEMPLATE.format(loc="https://other.example.org/nested.xml"))
    )

    assert loader.scraped_urls == ["https://other.example.org/nested.xml"]
    assert [el["loc"] for el in els] == ["https://example.com/page-1"]
