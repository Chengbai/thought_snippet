# Thought: Safe resource or temporary data cleanup. 
---
## Pattern:
1. **Resource** acquired before need be released even in error case
2. **Temporary data objects**, if created, need be cleaned even in the error case

### Example:
- DB connection
- Pytorch distributed training: `dist.init_process_group`
- PyTorch DataLoader: `dataloader.close()`

## Solution:
@contextmanager transforms a generator function into a context manager:
- Before yield = Setup (like __enter__)
- yield = Pause and run user code
- After yield = Cleanup (like __exit__)
```
@contextmanager
def safe_context():
    print("Setup")
    try:
        yield
    except Exception as e:
        print(f"Caught exception in context: {e}")
        # You can handle, log, or re-raise
        raise  # Re-raise the exception
    finally:
        print("Cleanup (always runs)")

# Usage
with safe_context():
    print("Doing work")
    raise ValueError("Something went wrong!")
    print("This never runs")

# Output:
# Setup
# Doing work
# Caught exception in context: Something went wrong!
# Cleanup (always runs)
# [ValueError is propagated]
```
