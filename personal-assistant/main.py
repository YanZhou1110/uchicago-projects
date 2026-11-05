from assistant import PersonalAssistant


def main():
    assistant = PersonalAssistant()

    print("Personal Assistant Chatbot")
    print("--------------------------")
    print("Type 'exit', 'quit', or 'bye' to stop.")
    print()
    print("Try:")
    print("- Add a meeting with Jason tomorrow at 3 PM about the NLP project.")
    print("- What events do I have tomorrow?")
    print("- Move my meeting with Jason to 4 PM.")
    print("- Delete the NLP study event.")
    print("- Search DuckDuckGo for latest NLP transformer models.")
    print()

    while True:
        user_input = input("You: ")
        response = assistant.respond(user_input)
        print(f"\nAssistant: {response}\n")

        if user_input.lower().strip() in ["exit", "quit", "bye"]:
            break


if __name__ == "__main__":
    main()
