# Hacktoberfest 2026

A month-long exploration of open-weight AI, VLA, and Physical AI.

## Goals
- Explore open-weight AI models
- Build VLA/Physical AI projects
- Participate in DEV Challenges
- Document experiments and learnings

## Projects

```
hacktoberfest-2026/
├── dev-challenge/          # Hacktoberfest DEV Challenge entries
│   ├── weekend-challenge/  # Shared Workbench Reset Assistant (local open-weight VLM)
│   ├── week-1/             # Picnic Packing with SmolVLA (paraphrased instructions in LIBERO)
│   └── week-2/             # Teaching the Picnic Robot My Words (SmolVLA fine-tune on a rented GPU)
├── global-hack-week/       # Global Hack Week: Hacktoberfest challenge write-ups
│   ├── 01-registration.md  # Registration challenges
│   └── github/             # GitHub challenges (Codespaces, repo management, Copilot)
├── devrelay/               # DevRelay usage notes and reviewed agent session transcripts
└── models/                 # Shared local model weights; excluded from Git
```

### DEV Challenge entries

| Folder | Theme | Project | Post |
| --- | --- | --- | --- |
| [weekend-challenge](dev-challenge/weekend-challenge/) | Build for a Friend: solve a real problem for someone you know | Shared Workbench Reset Assistant: a local open-weight VLM helps a coworker put the workbench back | [DEV](https://dev.to/johyeongseob/where-did-this-go-a-local-ai-workbench-reset-assistant-for-a-coworker-42co) |
| [week-1](dev-challenge/week-1/) | Touch Grass: get people off the screen and into the world | Can I ask a robot to pack my picnic in my own words? SmolVLA in LIBERO, no training | [DEV](https://dev.to/johyeongseob/can-i-ask-a-robot-to-pack-my-picnic-in-my-own-words-40f2) |
| [week-2](dev-challenge/week-2/) | Side Hustle: build something someone would actually pay for | Teaching the picnic robot my words: fine-tuning SmolVLA with one demo per item on a rented GPU | [DEV](https://dev.to/johyeongseob/for-the-price-of-a-coffee-my-picnic-robot-learned-how-i-talk-11n9) |

## Development Environment

| Component | Configuration |
| --- | --- |
| Host | Windows |
| CPU | Intel Core Ultra X7 358H; 16 logical CPUs visible in WSL |
| GPU | Intel Arc B390 GPU; Windows driver 32.0.101.8356 |
| RAM | 32 GB |
| Runtime | WSL with Ubuntu 24.04 LTS |
| Shell for project commands | Bash in Ubuntu |
| Python | 3.12.3 |
