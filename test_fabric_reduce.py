
import torch
from lightning.fabric import Fabric

def test_all_reduce():
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    
    t = torch.tensor([1.0])
    print(f"Before: {t}")
    result = fabric.all_reduce(t, reduce_op="sum")
    print(f"After: {t}")
    print(f"Result: {result}")
    
    if t.item() == result.item():
        print("Fabric.all_reduce seems to modify in-place or return same tensor (for 1 device)")
    else:
        print("Fabric.all_reduce returned new tensor")

if __name__ == "__main__":
    test_all_reduce()
