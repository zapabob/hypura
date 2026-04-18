use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Url;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

const PRODUCT_USER_AGENT: &str = "Hypura-KoboldCpp-WebSearch/0.8.0";
const MAX_CONTENT_CHARS: usize = 4_000;
const MAX_DESC_CHARS: usize = 600;

#[derive(Debug, Clone, Deserialize)]
pub struct WebSearchRequest {
    pub q: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebSearchResultItem {
    pub title: String,
    pub url: String,
    pub desc: String,
    pub content: String,
}

#[derive(Clone)]
pub struct WebSearchService {
    client: reqwest::Client,
}

impl WebSearchService {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(12))
            .user_agent(PRODUCT_USER_AGENT)
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()?;
        Ok(Self { client })
    }

    pub async fn search(&self, query: &str) -> Result<Vec<WebSearchResultItem>> {
        let query = query.trim();
        anyhow::ensure!(!query.is_empty(), "search query must not be empty");

        let html = self
            .client
            .get("https://html.duckduckgo.com/html/")
            .query(&[("q", query)])
            .send()
            .await
            .context("sending DuckDuckGo HTML search request")?
            .error_for_status()
            .context("DuckDuckGo HTML search returned an error status")?
            .text()
            .await
            .context("reading DuckDuckGo HTML response")?;

        let mut results = parse_duckduckgo_html(&html);
        results.truncate(3);

        for result in &mut results {
            let fetched = self.fetch_page_excerpt(&result.url).await.ok();
            result.content = fetched
                .filter(|text| !text.trim().is_empty())
                .unwrap_or_else(|| result.desc.clone());
        }

        Ok(results)
    }

    async fn fetch_page_excerpt(&self, url: &str) -> Result<String> {
        let html = self
            .client
            .get(url)
            .send()
            .await
            .with_context(|| format!("fetching page {url}"))?
            .error_for_status()
            .with_context(|| format!("page fetch failed for {url}"))?
            .text()
            .await
            .with_context(|| format!("reading page body for {url}"))?;
        Ok(extract_readable_text(&html, MAX_CONTENT_CHARS))
    }
}

pub fn parse_duckduckgo_html(html: &str) -> Vec<WebSearchResultItem> {
    let document = Html::parse_document(html);
    let result_selector = Selector::parse(".result").unwrap();
    let title_selector = Selector::parse("a.result__a").unwrap();
    let snippet_selector = Selector::parse(".result__snippet").unwrap();

    let mut results = Vec::new();
    for result in document.select(&result_selector) {
        let Some(anchor) = result.select(&title_selector).next() else {
            continue;
        };
        let href = anchor.value().attr("href").unwrap_or("").trim();
        let resolved_url = normalize_duckduckgo_result_url(href);
        if resolved_url.is_empty() {
            continue;
        }
        let title = normalize_text(&anchor.text().collect::<Vec<_>>().join(" "));
        if title.is_empty() {
            continue;
        }
        let desc = result
            .select(&snippet_selector)
            .next()
            .map(|node| normalize_text(&node.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();
        results.push(WebSearchResultItem {
            title,
            url: resolved_url,
            desc: truncate_text(&desc, MAX_DESC_CHARS),
            content: String::new(),
        });
    }
    results
}

pub fn extract_readable_text(html: &str, limit_chars: usize) -> String {
    let document = Html::parse_document(html);
    let body_selector = Selector::parse("body").unwrap();
    let raw = document
        .select(&body_selector)
        .next()
        .map(|body| body.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| document.root_element().text().collect::<Vec<_>>().join(" "));
    truncate_text(&normalize_text(&raw), limit_chars)
}

fn normalize_duckduckgo_result_url(raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }
    if let Ok(parsed) = Url::parse(raw) {
        if parsed
            .host_str()
            .map(|host| host.contains("duckduckgo.com"))
            .unwrap_or(false)
        {
            if let Some((_, value)) = parsed.query_pairs().find(|(key, _)| key == "uddg") {
                return value.into_owned();
            }
        }
        return parsed.to_string();
    }
    if raw.starts_with("//") {
        return format!("https:{raw}");
    }
    raw.to_string()
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_text(text: &str, limit_chars: usize) -> String {
    if text.chars().count() <= limit_chars {
        return text.to_string();
    }
    text.chars().take(limit_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_duckduckgo_html_results() {
        let html = r#"
        <div class="results">
          <div class="result">
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha">Example Alpha</a>
            <a class="result__snippet">Alpha snippet text.</a>
          </div>
          <div class="result">
            <a class="result__a" href="https://example.com/beta">Example Beta</a>
            <div class="result__snippet">Beta snippet text.</div>
          </div>
        </div>
        "#;

        let results = parse_duckduckgo_html(html);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Example Alpha");
        assert_eq!(results[0].url, "https://example.com/alpha");
        assert_eq!(results[0].desc, "Alpha snippet text.");
        assert_eq!(results[1].url, "https://example.com/beta");
    }

    #[test]
    fn extract_readable_text_normalizes_whitespace() {
        let html = "<html><body><main> Hello   world <p>from   Hypura</p></main></body></html>";
        assert_eq!(extract_readable_text(html, 100), "Hello world from Hypura");
    }
}
