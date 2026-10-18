# Getting Started with Copilot

**Repository:** [johyeongseob/skills-getting-started-with-github-copilot](https://github.com/johyeongseob/skills-getting-started-with-github-copilot)
**Pull request:** [#3 Improve student activity registration system](https://github.com/johyeongseob/skills-getting-started-with-github-copilot/pull/3)

A GitHub Skills exercise on the Mergington High School activities site (FastAPI backend, static HTML/JS frontend).

## Steps

1. **Hello Copilot.** Asked Copilot Chat to explain the project structure, ran the app, and used terminal inline chat (`Ctrl+I`) to create and publish the `accelerate-with-copilot` branch.
2. **Getting work done.**
   - Asked Chat with `#codebase` where a duplicate-registration bug came from; it pointed to `signup_for_activity` in `src/app.py`.
   - Fixed it by typing a comment and accepting Copilot's inline suggestion.
   - Used inline chat on the `activities` dictionary to add six new activities.
   - Generated the commit message with Copilot.
3. **Agent mode.** Asked Agent mode to show participants on each activity card, add a delete icon that unregisters a participant, and fix a bug where a new registration only appeared after a page refresh. The fix was one `fetchActivities()` call after a successful signup.
4. **Plan agent.** Planned backend tests in a separate `tests/` directory, then had Agent mode implement them with pytest and the Arrange-Act-Assert pattern and add `pytest` to `requirements.txt`. Result: 8 tests passing.
5. **Copilot in a pull request.** Opened PR #3, let Copilot write the PR summary, requested a Copilot code review, and merged.

## What I learned

- Each Copilot surface fits a different job: Chat to explore, inline suggestions to write, inline chat for scoped edits, Agent mode for multi-file features, and the Plan agent to agree on an approach before any edit.
- Agent mode asks for approval before running tools. In this codespace, running tests through VS Code's test tool and creating a virtual environment both stalled, so I skipped those prompts and verified in the browser or terminal instead.
- **Copilot's PR review found real problems in code that Agent mode wrote:** a stored XSS risk from inserting participant emails with `innerHTML` (high), no capacity check before signup (medium), and low contrast on the participant count (low). Generated code still needs an independent review step.

[Back to overview](../README.md)
