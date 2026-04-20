import SwinUNetR

if __name__ == "__main__":
    model = SwinUNetR.SwinUNETR([32, 32, 32], 3, 1)
    print(model)