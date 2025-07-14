class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def is_empty(self):
        return len(self.items) == 0

class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.insert(0, item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def is_empty(self):
        return len(self.items) == 0

def is_palindrome(s):
    stack = Stack()
    queue = Queue()
    
    # Process string (case insensitive, ignore spaces)
    processed = ''.join(c.lower() for c in s if c.isalnum())
    
    for char in processed:
        stack.push(char)
        queue.enqueue(char)
    
    while not stack.is_empty() and not queue.is_empty():
        if stack.pop() != queue.dequeue():
            return False
    
    return True

# Test cases
if __name__ == "__main__":
    test_cases = ["A man, a plan, a canal: Panama", "racecar", "hello", "No 'x' in Nixon"]
    for test in test_cases:
        print(f"'{test}': {is_palindrome(test)}")
