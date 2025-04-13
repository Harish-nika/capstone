import re
import os

def split_chat_to_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        chat = f.read()

    # Normalize special Unicode spaces from WhatsApp exports
    chat = chat.replace('\u202f', ' ').replace('\u200e', '')

    # WhatsApp datetime pattern
    pattern = r'(\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}\s?[ap]m\s- )'
    parts = re.split(pattern, chat)

    if len(parts) < 2:
        print("❌ No messages found. Check the formatting.")
        return []

    messages = []
    for i in range(1, len(parts), 2):
        timestamp = parts[i].strip()
        content = parts[i + 1].strip()
        messages.append(f"{timestamp} {content}")

    chunks = []
    chunk_id = 1

    for msg in messages:
        match = re.match(r'.* - ([^:]+): (.+)', msg)
        if match:
            user, text = match.groups()
            chunks.append(f"chunk{chunk_id}: {user}: {text}")
            chunk_id += 1
        else:
            print(f"[SKIPPED] Not a user message: {msg[:60]}...")

    return chunks

# ✅ Use corrected file path
file_path = "D:/codinghub/Content-moderator-image-1.3/test_text_vision_models/c1.txt"
chunks = split_chat_to_chunks(file_path)

for c in chunks:
    print(c)
