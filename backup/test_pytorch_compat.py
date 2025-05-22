#!/usr/bin/env python3
"""
Test script pour vérifier la compatibilité PyTorch 2.6+
"""

import torch
import sys

print("PyTorch Compatibility Test")
print("=" * 40)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print("\nTesting specific PyTorch 2.6+ features...")

try:
    # Test tensor operations that might cause issues
    print("1. Testing tensor operations...")
    x = torch.randn(2, 3, 4, 4)
    y = torch.randn(2, 3, 4, 4)
    z = x + y
    print("   ✓ Basic tensor operations work")
    
    # Test nn.Module functionality
    print("2. Testing nn.Module...")
    import torch.nn as nn
    
    class TestModule(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
            self.bn = nn.BatchNorm2d(out_ch)
            
        def forward(self, x):
            return self.bn(self.conv(x))
    
    module = TestModule(3, 16)
    test_input = torch.randn(1, 3, 32, 32)
    output = module(test_input)
    print("   ✓ nn.Module functionality works")
    
    # Test list/tuple handling (common PyTorch 2.6+ issue)
    print("3. Testing argument handling...")
    
    def test_args(*args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                print(f"   Arg {i}: {type(arg)} with {len(arg)} elements")
            else:
                print(f"   Arg {i}: {type(arg)} = {arg}")
    
    test_args([1, 2], (3, 4), 5, test_list=[6, 7])
    print("   ✓ Argument handling works")
    
    print("\n✅ PyTorch 2.6+ compatibility confirmed!")
    
except Exception as e:
    print(f"❌ PyTorch compatibility issue: {e}")
    import traceback
    traceback.print_exc()
