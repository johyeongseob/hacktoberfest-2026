# DevRelay

This directory contains DevRelay usage notes and agent session transcripts for Hacktoberfest projects. Project implementation belongs in `../dev-challenge/weekend-challenge/`. DevRelay itself is installed in the user environment, outside this repository.

## Current Status

- DevRelay 0.1.17 installed
- MLH login completed
- Installation and login sticker confirmed
- `devrelay sessions list` succeeded and returned no saved sessions (`[]`)
- MCP tool availability in the Codex chat has not yet been confirmed

## Directory Structure

- `sessions/`: reviewed session JSON files
- `sessions/weekend-challenge.json`: to be created after the Weekend Challenge work

Conversations are not saved automatically. Prepare and review the actual work transcript before uploading it. No session JSON has been created yet, to avoid uploading an empty example as a real session.

## Session Commands

DevRelay 0.1.17 is installed on Windows. Run the commands in this section in
Windows PowerShell. Python project execution uses WSL Ubuntu 24.04 and the
Linux virtual environment described in the [repository README](../README.md).
The Windows DevRelay installation does not establish a separate installation
inside Ubuntu.

List saved sessions in PowerShell:

```powershell
devrelay sessions list
```

Once the reviewed JSON file is ready, upload it from the repository root:

```powershell
devrelay sessions submit --title "Weekend Challenge: Build for a Friend" --file .\devrelay\sessions\weekend-challenge.json
```

Check the returned session ID or sharing link and review its publication status before linking it in the DEV submission. To embed a session in a DEV post, replace `SESSION_ID` below with the actual session ID:

```liquid
{% agent_session SESSION_ID %}
```

## Review Before Sharing

Before adding a transcript to Git or uploading it:

- Replace API keys, access tokens, and passwords with `[REDACTED_SECRET]`.
- Remove OAuth authorization codes and callback URLs.
- Anonymize personal information, including names and email addresses.
- Replace private absolute machine paths with repository-relative paths.
- Remove private data or replace it with sample data.

Include only user and assistant messages and work results selected for sharing. Preserve technical context while removing sensitive information. Keep raw transcripts outside Git; commit only reviewed, sanitized copies.

Write repository documents, code comments, and shared transcripts in American English (en-US).

## References

- [DevRelay](https://devrelay.com/)
- [Weekend Challenge](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)
