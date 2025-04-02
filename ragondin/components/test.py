import threading


class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()  # Ensures thread safety

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:  # First check (not thread-safe yet)
            with cls._lock:  # Prevents multiple threads from creating instances
                if cls not in cls._instances:  # Second check (double-checked locking)
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


# Example singleton class using the metaclass:
class MySingleton(metaclass=SingletonMeta):
    def __init__(self, value):
        self.value = value


# Testing the singleton:
singleton1 = MySingleton(10)
singleton2 = MySingleton(20)

print(singleton1.value)  # Expected: 10
print(singleton2.value)  # Expected: 10, not 20
print(singleton1 is singleton2)  # Expected: True


def create_instance(value):
    instance = MySingleton(value)
    print(f"Instance {id(instance)} with value {instance.value}")


threads = [threading.Thread(target=create_instance, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
