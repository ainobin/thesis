import os

BASE = os.path.dirname(os.path.abspath(__file__))

DIRS = [
    os.path.join(BASE, "data", "raw", "grade_a"),
    os.path.join(BASE, "data", "raw", "grade_b"),
    os.path.join(BASE, "data", "processed"),
    os.path.join(BASE, "src"),
    os.path.join(BASE, "models"),
]

def main() -> None:
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  \u2713 {os.path.relpath(d, BASE)}")
    print("Workspace ready.")

if __name__ == "__main__":
    main()
