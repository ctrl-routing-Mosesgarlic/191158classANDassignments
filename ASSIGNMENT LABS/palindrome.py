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
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def is_empty(self):
        return len(self.items) == 0

def is_palindrome(s):
    # Initialize stack and queue
    stack = Stack()
    queue = Queue()
    
    # Push all characters to stack and enqueue to queue
    for char in s:
        stack.push(char)
        queue.enqueue(char)
    
    # Compare characters from stack (LIFO) and queue (FIFO)
    while not stack.is_empty() and not queue.is_empty():
        stack_char = stack.pop()
        queue_char = queue.dequeue()
        
        # If characters don't match, it's not a palindrome
        if stack_char != queue_char:
            return False
    
    # If we've processed all characters without mismatch, it's a palindrome
    return True

# Main function to handle input and output
def main():
    # Read input
    s = input().strip()
    
    # Check if palindrome
    if is_palindrome(s):
        print(f"The word, {s}, is a palindrome.")
    else:
        print(f"The word, {s}, is not a palindrome.")

# Test the solution with sample cases
if __name__ == "__main__":
    # Test cases
    test_cases = ["racecar", "hello", "madam", "python", "level", "world"]
    
    print("Testing palindrome checker:")
    print("-" * 40)
    
    for test in test_cases:
        if is_palindrome(test):
            print(f"The word, {test}, is a palindrome.")
        else:
            print(f"The word, {test}, is not a palindrome.")
    
    print("\n" + "=" * 40)
    print("Enter your string to check:")
    main()
