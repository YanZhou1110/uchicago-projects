import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path


class CalendarTool:
    def __init__(self, file_path="calendar_events.json"):
        self.file_path = Path(file_path)
        self.events = self._load_events()

    def _load_events(self):
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _save_events(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)

    def create_event(self, title, date, time, notes=""):
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "title": title.strip(),
            "date": date,
            "time": time,
            "notes": notes.strip()
        }
        self.events.append(event)
        self._save_events()
        return event

    def retrieve_events(self, date=None, keyword=None):
        results = self.events

        if date:
            results = [event for event in results if event["date"] == date]

        if keyword:
            keyword = keyword.lower()
            results = [
                event for event in results
                if keyword in event["title"].lower() or keyword in event.get("notes", "").lower()
            ]

        results.sort(key=lambda x: (x["date"], x["time"]))
        return results

    def update_event(self, keyword, new_date=None, new_time=None, new_title=None, new_notes=None):
        matches = self.retrieve_events(keyword=keyword)

        if not matches:
            return None, "No matching event found."

        event = matches[0]

        if new_date:
            event["date"] = new_date
        if new_time:
            event["time"] = new_time
        if new_title:
            event["title"] = new_title.strip()
        if new_notes is not None:
            event["notes"] = new_notes.strip()

        self._save_events()
        return event, "Event updated successfully."

    def delete_event(self, keyword):
        matches = self.retrieve_events(keyword=keyword)

        if not matches:
            return None, "No matching event found."

        event = matches[0]
        self.events = [e for e in self.events if e["event_id"] != event["event_id"]]
        self._save_events()
        return event, "Event deleted successfully."


def parse_date(text):
    text = text.lower()
    today = datetime.now().date()

    if "today" in text:
        return today.isoformat()
    if "tomorrow" in text:
        return (today + timedelta(days=1)).isoformat()
    if "yesterday" in text:
        return (today - timedelta(days=1)).isoformat()

    month_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }

    pattern = r"\b(" + "|".join(month_map.keys()) + r")\s+(\d{1,2})(?:,\s*(\d{4}))?"
    match = re.search(pattern, text)

    if match:
        month = month_map[match.group(1)]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else today.year
        return datetime(year, month, day).date().isoformat()

    iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso_match:
        year, month, day = map(int, iso_match.groups())
        return datetime(year, month, day).date().isoformat()

    return None


def parse_time(text):
    text = text.lower()

    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        meridiem = match.group(3)

        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"

    return None


def clean_event_title(text):
    text = text.strip()

    remove_phrases = [
        "add", "create", "schedule", "put", "set up",
        "an event", "a meeting", "meeting", "event",
        "appointment", "calendar"
    ]

    for phrase in remove_phrases:
        text = re.sub(rf"\b{re.escape(phrase)}\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\b(today|tomorrow|yesterday)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(at|on|for)\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\s+\d{1,2}(,\s*\d{4})?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .")

    if not text:
        return "Untitled Event"

    return text[0].upper() + text[1:]


def extract_notes(text):
    match = re.search(r"\babout\b(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(" .")
    return ""
