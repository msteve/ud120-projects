#!/usr/bin/python3

import os
import pickle
import re


import os

def write_email_paths(base_dir, output_file):
    with open(output_file, "w") as out:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start="..")  # relative to project root
                out.write(rel_path + "\n")

# Example usage:
# write_email_paths("../maildir/shackleton-s", "from_sara.txt")
# write_email_paths("../maildir/germani-c", "from_chris.txt")


from parse_out_email_text import parseOutText

# Paths to Sara and Chris's email folders
sara_path = os.path.join('..', 'maildir', 'sara')
chris_path = os.path.join('..', 'maildir', 'chris')

# Files that contain the paths to Sara and Chris's emails
from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

# Store the processed text and labels
word_data = []
from_data = []

# List of words to remove from email text
remove_words = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        path = os.path.join('..', path.strip())
        if not os.path.isfile(path):
            continue
        with open(path, "r", errors='ignore') as email:
            text = parseOutText(email)
            for word in remove_words:
                text = text.replace(word, "")
            word_data.append(text)
            from_data.append(0 if name == "sara" else 1)

from_sara.close()
from_chris.close()

print("emails processed")

# Save the processed word data and author labels
with open("your_word_data.pkl", "wb") as f:
    pickle.dump(word_data, f)

with open("your_email_authors.pkl", "wb") as f:
    pickle.dump(from_data, f)