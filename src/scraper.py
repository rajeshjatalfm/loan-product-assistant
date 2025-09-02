import requests
from bs4 import BeautifulSoup
import json,os


def scrape_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    # Extract headings, paragraphs, tables
    content = []
    for sec in soup.find_all(["p", "li", "table", "h2", "h3"]):
        content.append(sec.get_text(strip=True))
    return " ".join(content)

def main():
    urls = [
        "https://bankofmaharashtra.in/advances",
        "https://bankofmaharashtra.in/maha-super-flexi-housing-loan-scheme",
        "https://bankofmaharashtra.in/loan-against-property",
        "https://bankofmaharashtra.in/personal-banking/loans/personal-loan",
        "https://bankofmaharashtra.in/mahabank-vehicle-loan-scheme-for-second-hand-car",
        "https://bankofmaharashtra.in/maha-super-housing-loan-scheme-for-construction-acquiring",
        "https://bankofmaharashtra.in/maha-super-housing-loan-scheme-for-purchase-plot-construction-thereon"
    ]
    data = {}
    for url in urls:
        data[url] = scrape_page(url)

    os.makedirs("../loan/data_scraped", exist_ok=True)
    with open("../loan/data_scraped/raw_data.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
