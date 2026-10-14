# Global Hack Week: Hacktoberfest

My participation in [Global Hack Week: Hacktoberfest](https://ghw.mlh.com/events/open-source), a free online week run by MLH from **Oct 9 to Oct 15, 2026** as part of Hacktoberfest 2026. The week's theme was open-weight models; its partner challenges came from GitHub, Tiger Data, and SpaceX (Cursor and Grok Bot).

I focused on the **GitHub challenges**, which needed only my existing GitHub account and taught skills I use day to day, plus the week's **registration challenges**.

## Submitted challenges

| # | Challenge | Repository | Status | Notes |
|---|---|---|---|---|
| 1 | [Registration challenges](challenges/01-registration.md): Register and Check In, Address Form, GitHub Student Developer Pack, Sign up for a Cursor Pro Account | — | Partly approved | Cursor Pro is now "Premium" in the X Subscription |
| 2 | [Code with Codespaces](challenges/02-code-with-codespaces.md) | [skills-code-with-codespaces](https://github.com/johyeongseob/skills-code-with-codespaces) | Approved | Custom dev container |
| 3 | [Intro to Repository Management](challenges/03-repository-management.md) | [skills-introduction-to-repository-management](https://github.com/johyeongseob/skills-introduction-to-repository-management) | Approved | Rulesets, CODEOWNERS, security setup |
| 4 | [Getting Started with Copilot](challenges/04-getting-started-with-copilot.md) | [skills-getting-started-with-github-copilot](https://github.com/johyeongseob/skills-getting-started-with-github-copilot) | Submitted | Ask, inline, Agent, and Plan modes; [PR #3](https://github.com/johyeongseob/skills-getting-started-with-github-copilot/pull/3) |
| 5 | [Use GitHub Copilot to complete any of this week's challenges](challenges/05-use-copilot-for-a-challenge.md) | [skills-getting-started-with-github-copilot](https://github.com/johyeongseob/skills-getting-started-with-github-copilot) | Submitted | Copilot used across challenge 4 |
| 6 | [Use GitHub Copilot to Build a Simple Application](challenges/06-copilot-simple-application.md) | [copilot-adventure-game](https://github.com/johyeongseob/copilot-adventure-game) | Submitted | CLI choose-your-own-adventure game |

## Livestreams

- **Opening Ceremony (Oct 9):** checked in with the on-screen code. It counted toward three Hacktoberfest stickers: the GHW livestream sticker, the general livestream sticker, and a secret "Crashtoberfest" sticker given during a check-in outage.
- I watched with Chrome's built-in **Live Caption** and its live translation into Korean, which works on Twitch, where there are no subtitles.

## Skipped challenges

| Partner | Why I skipped |
|---|---|
| SpaceX (Cursor and Grok Bot challenges 1–7) | No-code product tutorials that need extra accounts; little overlap with my work. I only did the Cursor Pro sign-up, which is a registration challenge. |
| Tiger Data (Tiger Cloud, hypertables, continuous aggregates) | Requires a Tiger Cloud account; not related to my current work. |

## Takeaways

- **Dev containers** turn "it works on my machine" into a one-click, reproducible environment.
- **Repository hygiene** (rulesets, CODEOWNERS, contribution guides, Dependabot, CodeQL, a security policy) is what makes a project safe to open to collaborators.
- **Copilot is a set of tools, not one feature:** Chat to explore, inline suggestions to write, Agent mode to implement across files, the Plan agent to design first, and PR review as a second check.
- **An independent review step still matters.** In challenge 4, Copilot's PR review flagged a stored XSS issue that Agent mode had introduced in the same PR.
