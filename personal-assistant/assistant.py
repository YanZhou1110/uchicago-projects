import re

from calendar_tool import (
    CalendarTool,
    parse_date,
    parse_time,
    clean_event_title,
    extract_notes
)
from search_tool import SearchTool


class PersonalAssistant:
    def __init__(self):
        self.calendar = CalendarTool()
        self.search_tool = SearchTool()
        self.last_event_keyword = None
        self.last_date = None
        self.last_search_query = None

    def respond(self, user_input):
        text = user_input.strip()
        lowered = text.lower()

        if lowered in ["quit", "exit", "bye"]:
            return "Goodbye! Your calendar events are saved locally."

        if self._is_search_request(lowered):
            return self._handle_search(text)

        if self._is_create_request(lowered):
            return self._handle_create(text)

        if self._is_retrieve_request(lowered):
            return self._handle_retrieve(text)

        if self._is_update_request(lowered):
            return self._handle_update(text)

        if self._is_delete_request(lowered):
            return self._handle_delete(text)

        return (
            "I can help with calendar events or web search.\n"
            "Examples:\n"
            "- Add a meeting with Jason tomorrow at 3 PM about the NLP project.\n"
            "- What events do I have tomorrow?\n"
            "- Move my meeting with Jason to 4 PM.\n"
            "- Delete the NLP study event.\n"
            "- Search DuckDuckGo for transformer models in NLP."
        )

    def _is_search_request(self, lowered):
        keywords = ["search", "look up", "find information", "google", "duckduckgo", "web"]
        return any(keyword in lowered for keyword in keywords)

    def _is_create_request(self, lowered):
        keywords = ["add", "create", "schedule", "set up", "put"]
        return any(keyword in lowered for keyword in keywords)

    def _is_retrieve_request(self, lowered):
        keywords = ["what events", "show", "retrieve", "schedule", "calendar", "what do i have"]
        return any(keyword in lowered for keyword in keywords)

    def _is_update_request(self, lowered):
        keywords = ["move", "change", "update", "reschedule", "edit"]
        return any(keyword in lowered for keyword in keywords)

    def _is_delete_request(self, lowered):
        keywords = ["delete", "cancel", "remove"]
        return any(keyword in lowered for keyword in keywords)

    def _handle_search(self, text):
        query = re.sub(
            r"^(search|look up|find information about|find|google|duckduckgo|search duckduckgo for)\s+",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

        if not query:
            query = self.last_search_query

        if not query:
            return "What would you like me to search for?"

        self.last_search_query = query
        results = self.search_tool.search(query)

        response = [f"Search results for: {query}"]
        for i, result in enumerate(results, start=1):
            response.append(
                f"\n{i}. {result['title']}\n"
                f"   {result['body']}\n"
                f"   {result['href']}"
            )

        return "\n".join(response)

    def _handle_create(self, text):
        date = parse_date(text)
        time = parse_time(text)
        notes = extract_notes(text)
        title = clean_event_title(text)

        if not date:
            return "I could not detect the date. Try saying something like 'tomorrow' or 'May 15'."

        if not time:
            return "I could not detect the time. Try saying something like '3 PM' or '15:00'."

        event = self.calendar.create_event(title, date, time, notes)
        self.last_event_keyword = title
        self.last_date = date

        return (
            "Event created successfully.\n"
            f"ID: {event['event_id']}\n"
            f"Title: {event['title']}\n"
            f"Date: {event['date']}\n"
            f"Time: {event['time']}\n"
            f"Notes: {event['notes'] or 'None'}"
        )

    def _handle_retrieve(self, text):
        date = parse_date(text)

        if not date and any(word in text.lower() for word in ["it", "that day", "same day"]):
            date = self.last_date

        events = self.calendar.retrieve_events(date=date)

        if not events:
            if date:
                return f"No events found for {date}."
            return "No events found."

        self.last_date = date

        response = ["Here are your events:"]
        for event in events:
            response.append(
                f"\n- [{event['event_id']}] {event['title']}\n"
                f"  Date: {event['date']}\n"
                f"  Time: {event['time']}\n"
                f"  Notes: {event['notes'] or 'None'}"
            )

        return "\n".join(response)

    def _handle_update(self, text):
        new_date = parse_date(text)
        new_time = parse_time(text)

        keyword = self._extract_event_keyword(text)
        if not keyword and self.last_event_keyword:
            keyword = self.last_event_keyword

        if not keyword:
            return "Which event should I update? Please include part of the event title."

        new_notes = None
        if "notes" in text.lower():
            note_match = re.search(r"(?:notes?|note)\s+(?:to|for|as)?\s*(.+)", text, re.IGNORECASE)
            if note_match:
                new_notes = note_match.group(1).strip()

        event, message = self.calendar.update_event(
            keyword=keyword,
            new_date=new_date,
            new_time=new_time,
            new_notes=new_notes
        )

        if not event:
            return message

        self.last_event_keyword = event["title"]
        self.last_date = event["date"]

        return (
            f"{message}\n"
            f"Title: {event['title']}\n"
            f"Date: {event['date']}\n"
            f"Time: {event['time']}\n"
            f"Notes: {event['notes'] or 'None'}"
        )

    def _handle_delete(self, text):
        keyword = self._extract_event_keyword(text)

        if not keyword and self.last_event_keyword:
            keyword = self.last_event_keyword

        if not keyword:
            return "Which event should I delete? Please include part of the event title."

        event, message = self.calendar.delete_event(keyword)

        if not event:
            return message

        return (
            f"{message}\n"
            f"Deleted event: {event['title']} on {event['date']} at {event['time']}"
        )

    def _extract_event_keyword(self, text):
        cleaned = text.lower()

        remove_words = [
            "move", "change", "update", "reschedule", "edit",
            "delete", "cancel", "remove",
            "my", "the", "event", "meeting", "appointment",
            "to", "at", "on", "for", "notes", "note"
        ]

        for word in remove_words:
            cleaned = re.sub(rf"\b{word}\b", " ", cleaned)

        cleaned = re.sub(r"\b(today|tomorrow|yesterday)\b", " ", cleaned)
        cleaned = re.sub(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")

        if len(cleaned) < 2:
            return None

        return cleaned
