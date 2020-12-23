# SimpleLearn

A python package to simplify data modeling.

## Build Locally

You can build your most recent changes by running the following command from the root directory:
```
$ pip install -e ."[devel]"
```

You can then import the package into your own code:
```python
# With recent changes
import simple_learn
```

## Contributing

There is a lot to do so contributions are really appreciated! This is a great project for early stage developers to work with.

To begin it is recommended starting with issues labelled [good first issue](https://github.com/skekre98/simple_learn/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).


How to get started:

1. Fork the simple_learn repo.
2. Create a new branch in you current repo from the 'main' branch with issue label.
3. 'Check out' the code with Git or [GitHub Desktop](https://desktop.github.com/)
4. Check [contributing.md](CONTRIBUTING.md)
5. Prior to pushing your changes, run the following command to format/lint your code(The CI pipeline will fail if standards are not met) followed by creating a Pull Request (PR) on simple_learn.
```bash
$ pre-commit run --all-files
```
