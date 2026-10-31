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

## License

Copyright 2026 Hyeongseob Jo. The code and writing in this repository are
licensed under the [Apache License 2.0](LICENSE).

Models, data and simulators used here keep their own licenses:

| Used | License |
| --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot), [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero), [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero) | Apache-2.0 |
| [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) (Weekend Challenge) | Apache-2.0 |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (scenes and assets in the videos); [LIBERO datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) | MIT; Apache-2.0 |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 |
| [`weekend-challenge/data`](dev-challenge/weekend-challenge/data) | Its own [LICENSE.txt](dev-challenge/weekend-challenge/data/LICENSE.txt) (MIT) |
