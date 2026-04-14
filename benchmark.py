import time
import torch
import requests
from src.student import StudentModel
from src.teacher import TeacherModel

# model size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def compare_size():
    print("=" * 40)
    print("MODEL SIZE")
    print("=" * 40)

    teacher = TeacherModel()
    student = StudentModel()

    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)

    print(f"Teacher parameters : {teacher_params:,}")
    print(f"Student parameters : {student_params:,}")
    print(f"Compression ratio  : {teacher_params / student_params:.1f}x smaller")

# inference speed

def measure_speed(model, num_runs=100):
    model.eval()
    # dummy input simulating one tokenized sentence
    input_ids      = torch.randint(0, 30522, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    # warmup — first runs are always slower, don't count them
    for _ in range(10):
        with torch.no_grad():
            model(input_ids, attention_mask)

    # actual benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(input_ids, attention_mask)
    end = time.time()

    avg_ms = (end - start) / num_runs * 1000
    return avg_ms

def compare_speed():
    print("=" * 40)
    print("INFERENCE SPEED")
    print("=" * 40)

    teacher = TeacherModel()
    student = StudentModel()

    teacher_ms = measure_speed(teacher)
    student_ms = measure_speed(student)

    print(f"Teacher avg inference : {teacher_ms:.2f} ms")
    print(f"Student avg inference : {student_ms:.2f} ms")
    print(f"Speedup              : {teacher_ms / student_ms:.1f}x faster")

# api throughput
def measure_throughput(url, num_requests=100):
    print(f"Benchmarking {url}")
    payload = {"text": "This movie was absolutely fantastic!"}

    start = time.time()
    for _ in range(num_requests):
        requests.post(url, json=payload)
    end = time.time()

    total_time = end - start
    throughput = num_requests / total_time
    print(f"  Total time  : {total_time:.2f}s")
    print(f"  Throughput  : {throughput:.1f} requests/sec")
    return throughput

# then in __main__:
if __name__ == "__main__":
    compare_size()
    print()
    compare_speed()
    print()
    measure_throughput("http://localhost:8001/predict")  # teacher
    measure_throughput("http://localhost:8000/predict")  # student