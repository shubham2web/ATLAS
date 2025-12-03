import re

# Read server.py
with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all emoji characters (Unicode ranges for emojis)
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002700-\U000027BF"  # Dingbats
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U00002600-\U000026FF"  # Miscellaneous Symbols
    u"\U0001F700-\U0001F77F"  # Alchemical Symbols
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    u"\U00002300-\U000023FF"  # Miscellaneous Technical
    "]+", flags=re.UNICODE)

content_cleaned = emoji_pattern.sub('', content)

# Write back
with open('server.py', 'w', encoding='utf-8') as f:
    f.write(content_cleaned)

print(f"Removed emojis from server.py")
