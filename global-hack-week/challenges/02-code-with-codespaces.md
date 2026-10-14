# Code with Codespaces

**Repository:** [johyeongseob/skills-code-with-codespaces](https://github.com/johyeongseob/skills-code-with-codespaces)
**Status:** Approved (exercise issue closed as completed)

A GitHub Skills exercise on cloud development environments.

## Steps

1. **Start a codespace and push code.** Opened a codespace on the default "universal" image, checked tool versions (Node, .NET, Python, gh), and pushed a fix to `src/hello.py`.
2. **Use a custom image.** Added `.devcontainer/devcontainer.json` with the `mcr.microsoft.com/devcontainers/base:debian` image and rebuilt the container. The tools from the default image disappeared, which showed the environment is now defined by the repository.
3. **Add features.** Extended the configuration with the Python 3.10 dev container feature, the Python and Python Debugger extensions, and a `postCreateCommand` script that installs `sl`.
4. **Test as a new developer.** Deleted the codespace and created a new one; the environment came back fully configured.

## What I learned

- A codespace is a dev container hosted by GitHub, defined by `devcontainer.json` in the repository.
- Rebuilds take a few minutes because the image, features, extensions, and lifecycle scripts all run again; prebuilds solve that in real projects.
- Delete codespaces when finished: stopped codespaces still count against storage quota.

[Back to overview](../README.md)
