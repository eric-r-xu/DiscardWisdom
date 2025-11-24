# path: overlay_dynamic_text.py
import tkinter as tk
import random

PHRASES = [
    "I'm always on top 👀",
    "Staying above the fold ✨",
    "Hello from Tkinter 🐍",
    "Overlay alive — no sleep! ☕",
    "Updating every 5s ⏱️",
    "You got this 🚀",
    "Focus mode engaged 🎯",
    "Stay sharp 🔪",
    "Ship it! 📦",
    "Keep calm and code on 💻",
]

def main():
    root = tk.Tk()
    root.title("Overlay GUI")
    root.attributes("-topmost", True)  # keep window above others
    root.geometry("360x120")

    label_var = tk.StringVar(value=random.choice(PHRASES))
    label = tk.Label(root, textvariable=label_var, wraplength=320, justify="center")
    label.pack(pady=16, padx=12)

    tk.Button(root, text="Close", command=root.destroy).pack()

    def update_text():
        # why: avoid showing the same phrase twice in a row
        current = label_var.get()
        choices = [p for p in PHRASES if p != current] or PHRASES
        label_var.set(random.choice(choices))
        root.after(5000, update_text)

    # start the periodic update
    root.after(5000, update_text)

    root.mainloop()

if __name__ == "__main__":
    main()
