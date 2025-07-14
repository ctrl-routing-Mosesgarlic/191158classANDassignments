# Palindrome Check Using Stack and Queue (HackerRank Day 18 Inspired)

## Problem Description

A **palindrome** is a word, phrase, number, or other sequence of characters which reads the same backward as forward.

Your task is to determine if a given string `s` is a palindrome by using both a **stack** and a **queue**.

---

## How to Solve

- Push each character of the string onto a **stack**.
- Enqueue each character of the string into a **queue**.
- Then, pop characters from the stack and dequeue characters from the queue one by one.
- Compare the characters obtained from the stack and queue.
- If all corresponding characters match, the string is a palindrome.

---

## Input Format

- A single line containing the string `s`.
- The string consists of lowercase English letters only.

---

## Output Format

- Print `"The word, s, is a palindrome."` if the string is a palindrome.
- Otherwise, print `"The word, s, is not a palindrome."`.

---

## Constraints

- `1 <= len(s) <= 50`

---

## Sample Input
