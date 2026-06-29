from langchain_core.messages import HumanMessage;
from langchain_groq import ChatGroq;
from dotenv import load_dotenv;

load_dotenv();

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
);

print("=== LangChain Invoke & Stream Demo ===");

while True:

    print("\nChoose Response Mode:");

    print("1. Invoke (Wait for complete response)");

    print("2. Stream (Token by token response)");

    print("3. Exit");

    choice = input("\nEnter Choice: ").strip();

    if choice == "3":

        print("\n👋 Goodbye!");

        break;

    if choice not in ("1", "2"):

        print("\n❌ Invalid Choice!");

        continue;

    user_input = input("\n👤 You: ").strip();

    if user_input.lower() in ("exit", "quit"):

        print("\n👋 Goodbye!");

        break;

    if choice == "1":

        print("\n🤖 AI:\n");

        try:

            response = llm.invoke([HumanMessage(content=user_input)]);

            print(response.content);

        except Exception as e:

            print(f"\n❌ Error: {e}");

    elif choice == "2":

        print("\n🤖 AI:\n");

        try:

            for chunk in llm.stream([HumanMessage(content=user_input)]):

                print(chunk.content, end="", flush=True);

        except Exception as e:

            print(f"\n❌ Error: {e}");

    else:

        print("\n❌ Invalid Choice!");