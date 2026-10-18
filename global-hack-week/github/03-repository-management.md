# Intro to Repository Management

**Repository:** [johyeongseob/skills-introduction-to-repository-management](https://github.com/johyeongseob/skills-introduction-to-repository-management)

A GitHub Skills exercise on preparing a repository (the Mergington High School activities site) for safe collaboration.

## Steps

1. **Protect your code.** Created a branch ruleset "Protect main" (restrict deletions, require a pull request, require Code Owner review, block force pushes) and added a `.gitignore` through a pull request.
2. **Prepare to collaborate.** Added `CONTRIBUTING.md` and `CODEOWNERS` on the `prepare-to-collaborate` branch.
3. **Foster healthy growth.** Added `CODE_OF_CONDUCT.md`, plus bug report and feature request issue templates on a separate `add-issue-templates` branch.
4. **Prepare for the inevitable.** Enabled Dependabot alerts, security updates, and `pip` version updates; enabled CodeQL default code scanning with Copilot Autofix; enabled private vulnerability reporting; added `SECURITY.md`.
5. **Release.** Merged both pull requests.

## What I learned

- The ruleset immediately blocked a direct commit to `main`, so every change had to go through a pull request.
- GitHub Skills checks branch names exactly; GitHub's default `<user>-patch-1` name made one step fail until I renamed the branch.
- End-of-line comments are not valid in `.gitignore`. The exercise's sample uses them, so a real project should put comments on their own lines.

[Back to overview](../README.md)
