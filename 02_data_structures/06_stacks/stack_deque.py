from collections import deque

stack = deque([10, 20, 30])
print(f"  Initial deque: {list(stack)}")

stack.append(100)
stack.append(200)
stack.append(300)

print(f"  Stack (deque): {list(stack)}")

item = stack.pop()
print(f"  Popped from deque: {item}")
print(f"  After pop: {list(stack)}")

# Append (Right Side) – Stack
stack.append(40)
print(f"  After append(40): {list(stack)}")

# Append Left – Queue
stack.appendleft(5)
print(f"  After appendleft(5): {list(stack)}")

# Pop (Right Side)
right = stack.pop()
print(f"  Popped from right: {right}")
print(f"  After pop(): {list(stack)}")

# Pop (Left Side)
left = stack.popleft()
print(f"  Popped from left: {left}")
print(f"  After popleft(): {list(stack)}")


print(f"  Front item: {stack[0]}")
print(f"  Rear item: {stack[-1]}")


# Rotate Elements
stack.rotate(1)  # Moves last element to front
print(f"  After rotate(1): {list(stack)}")

stack.rotate(-2)  # Moves front two elements to end
print(f"  After rotate(-2): {list(stack)}")

# Clear Deque
stack.clear()
print(f"  After clear(): {list(stack)}")



