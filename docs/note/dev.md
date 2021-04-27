# Development

The TorchShard project welcomes your expertise and enthusiasm!
If you are interested in torchshard, you are welcome to help to make it better and more professional.

## Test Cases

For major changes, please write full tests and make sure all the changes pass them.

Test cases for a new core feature should:
- have its non-parallel implementation and parallel implementation
- have a test with DDP to wrap up the target layer or module
- pass forward equality checks, including layer outputs, loss values, and any other kinds of intermediate outputs
- pass backward equality checks, including outputs and gradients
- pass all the checks with the threshold value `1e-6`.

The [tests](../../tests) folder holds the test scripts.

## Install Locally for Test

```bash
# under the project root
python3 setup.py install && rm -fr ./build ./dist ./torchshard.egg-info 
```

<p><br/></p>

<p>&#10141; Back to the <a href="../">main page</a></p>
