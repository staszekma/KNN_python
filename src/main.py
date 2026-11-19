import sys

try:
    from KNN_python.src.Chart import Chart2D, Chart3D, Chart2D_c
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def main():

    Chart2D([1, 2],[2, 3])


if __name__ == "__main__":
    main()