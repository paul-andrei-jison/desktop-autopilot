import ollama

MODEL = "llama3.2"
PROMPT = "Hello! Introduce yourself as Bart, my AI assistant."


def main():
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    print(response["message"]["content"].strip())


if __name__ == "__main__":
    main()

