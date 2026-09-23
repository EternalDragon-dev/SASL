from __future__ import annotations


class WordBuffer:
    """Tracks the current in-progress word and the committed words."""

    def __init__(self) -> None:
        self.current_word = ""
        self.committed_words: list[str] = []

    def add_letter(self, letter: str | None) -> None:
        if not letter:
            return
        self.current_word += str(letter).upper()

    def commit_current_word(self) -> str:
        if not self.current_word:
            return ""
        self.committed_words.append(self.current_word)
        word = self.current_word
        self.current_word = ""
        return word

    def clear(self) -> None:
        self.current_word = ""
        self.committed_words.clear()
