import numpy as np
import torch


class SyncArray:
    def __init__(self, array, sync_callback):
        self.array = array
        self.sync_callback = sync_callback  # Function to call on modification

    def __getitem__(self, idx):
        return self.array[idx]

    def __setitem__(self, idx, value):
        self.array[idx] = value
        self.sync_callback()  # Sync to PyTorch whenever NumPy array is modified

    def __array__(self):
        return self.array

    def __repr__(self):
        return repr(self.array)


class DualArray:
    def __init__(self, data):
        # Initialize from either a NumPy array or a PyTorch tensor
        if isinstance(data, np.ndarray):
            self.numpy_array = SyncArray(data, self._sync_from_numpy)
            self.torch_tensor = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            self.torch_tensor = data
            self.numpy_array = SyncArray(
                data.cpu().detach().numpy(), self._sync_from_numpy
            )
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor")

    # Method to update both arrays when one is modified
    def _sync_from_numpy(self):
        self.torch_tensor = torch.from_numpy(self.numpy_array.array).float()

    def _sync_from_torch(self):
        self.numpy_array = SyncArray(
            self.torch_tensor.cpu().detach().numpy(), self._sync_from_numpy
        )

    @property
    def numpy(self):
        return self.numpy_array

    @numpy.setter
    def numpy(self, value):
        if isinstance(value, np.ndarray):
            # Update the NumPy array and sync with the torch tensor
            self.numpy_array = SyncArray(value, self._sync_from_numpy)
            self._sync_from_numpy()  # Sync to PyTorch
        else:
            raise TypeError("Assigned value must be a NumPy array")

    @property
    def torch(self):
        return self.torch_tensor

    @torch.setter
    def torch(self, value):
        if isinstance(value, torch.Tensor):
            self.torch_tensor = value
            self._sync_from_torch()
        else:
            raise TypeError("Assigned value must be a PyTorch tensor")

    # Allow NumPy operations by converting to NumPy implicitly
    def __array__(self):
        return self.numpy_array.array

    # Allow PyTorch operations by converting to PyTorch implicitly
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Find the DualArray instance in the args, replace it with the internal torch tensor
        new_args = []
        for arg in args:
            if isinstance(arg, DualArray):
                new_args.append(arg.torch_tensor)
            else:
                new_args.append(arg)

        # Delegate the function call to the torch tensor
        return func(*new_args, **kwargs)

    # Arithmetic operations (addition, subtraction, multiplication, division)
    def __add__(self, other):
        if isinstance(other, DualArray):
            # Perform addition on both NumPy array and Torch tensor
            new_numpy = self.numpy_array.array + other.numpy_array.array
        else:
            # Perform addition with scalar or other types
            new_numpy = self.numpy_array.array + other

        # Return a new DualArray object with the results
        return DualArray(new_numpy)

    def __sub__(self, other):
        if isinstance(other, DualArray):
            new_numpy = self.numpy_array.array - other.numpy_array.array
        else:
            new_numpy = self.numpy_array.array - other

        return DualArray(new_numpy)

    def __mul__(self, other):
        if isinstance(other, DualArray):
            new_numpy = self.numpy_array.array * other.numpy_array.array
        else:
            new_numpy = self.numpy_array.array * other

        return DualArray(new_numpy)

    def __truediv__(self, other):
        if isinstance(other, DualArray):
            new_numpy = self.numpy_array.array / other.numpy_array.array
        else:
            new_numpy = self.numpy_array.array / other

        return DualArray(new_numpy)

    # Ensure that when printed, both representations are shown
    def __repr__(self):
        return f"DualArray:\nNumPy Array:\n{self.numpy_array}\nTorch Tensor:\n{self.torch_tensor}"


# Some testing
if __name__ == "__main__":
    # Initialize with a NumPy array
    np_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    dual = DualArray(np_data)
    print(dual)

    # Modify the NumPy array
    dual.numpy[0, 0] = 10  # This should trigger the sync
    print("\nAfter modifying NumPy array:")
    print(dual)

    # Torch operation: sum after modification
    torch_sum_after_mod = torch.sum(dual)
    print(f"\nTorch sum after modification: {torch_sum_after_mod}")

    # Numpy operation: sum after modification
    numpy_sum_after_mod = np.sum(dual)
    print(f"\nNumpy sum after modification: {numpy_sum_after_mod}")

    dual *= 0
    print(dual)

    dual.numpy = np.array([[2.0, 2.0], [5.0, 7.0]])
    print(dual)
