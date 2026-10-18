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
