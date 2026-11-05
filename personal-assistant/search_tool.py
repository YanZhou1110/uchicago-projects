try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


class SearchTool:
    def __init__(self, max_results=5):
        self.max_results = max_results

    def search(self, query):
        if DDGS is None:
            return [
                {
                    "title": "DDGS package is not installed",
                    "href": "",
                    "body": "Run: pip install ddgs"
                }
            ]

        results = []

        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=self.max_results):
                    results.append({
                        "title": item.get("title", "No title"),
                        "href": item.get("href", ""),
                        "body": item.get("body", "")
                    })
        except Exception as e:
            return [
                {
                    "title": "Search error",
                    "href": "",
                    "body": str(e)
                }
            ]

        return results
