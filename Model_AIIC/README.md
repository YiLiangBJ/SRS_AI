# Channel Separator Models for SRS Multi-Port Estimation

## 🎯 Quick Start

### Test Models

```bash
cd Model_AIIC

# Test all models
python channel_separator.py

# Test with data generator integration
python test_separator.py --model mlp --epochs 10
python test_separator.py --model residual --epochs 20
python test_separator.py --model hinted --epochs 10
```

## 📁 Files

- `channel_separator.py` - Model implementations (3 variants)
- `test_separator.py` - Integration test with data generator
- `INTEGRATION_GUIDE.md` - Detailed integration guide (Chinese)

## 🏗️ Models

### 1. SimpleMLPSeparator
- **Parameters**: ~20K
- **Best for**: Fast training, simple baseline

### 2. ResidualRefinementSeparator  
- **Parameters**: ~15K
- **Best for**: Iterative refinement with residual correction

### 3. PositionHintedSeparator
- **Parameters**: ~25K  
- **Best for**: Stable training with position hints

## 📊 Problem

$$y = \sum_{p \in P} \text{circshift}(h_p, p) + noise$$

**Goal**: Separate mixed signal into individual shifted components

## 🔗 Integration

See `INTEGRATION_GUIDE.md` for detailed integration steps with existing codebase.

Quick summary:
1. Add to `data_generator.py` - data generation
2. Modify `trainMLPmmse.py` - training loop  
3. Update `evaluate_performance.py` - evaluation

## ✅ Next Steps

1. Run quick tests (above)
2. Read `INTEGRATION_GUIDE.md`
3. Choose integration approach
4. Start with data generation modification
