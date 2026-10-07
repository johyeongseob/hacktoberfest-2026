# Local Demo Setup and Usage

The demo uses the open-weighted SmolVLM2 2.2B Q8 model through a local llama.cpp
server. Run both services in WSL and keep both terminals open. No hosted inference API key is required.

These instructions assume that the project virtual environment, Hugging Face CLI,
and configured llama.cpp checkout already exist. The browser demo uses llama.cpp
for inference and requires Streamlit and Pillow in the Python environment.

Download the browser demo's Q8 model and vision projector from the repository root:

```bash
source ~/.venvs/hacktoberfest-2026/bin/activate
cd /path/to/hacktoberfest-2026
hf download ggml-org/SmolVLM2-2.2B-Instruct-GGUF \
  SmolVLM2-2.2B-Instruct-Q8_0.gguf \
  mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf \
  --local-dir models/SmolVLM2-2.2B-Instruct-GGUF
```

First, build the server using the existing llama.cpp checkout and install the UI:

```bash
source ~/.venvs/hacktoberfest-2026/bin/activate
cmake --build ~/tools/llama.cpp/build --config Release -j 8 --target llama-server
python -m pip install 'streamlit>=1.40,<2'
python -m pip check
```

Terminal 1: start the model server and wait until it reports that it is listening:

```bash
cd /path/to/hacktoberfest-2026/dev-challenge/weekend-challenge
bash start_model_server.sh
```

Terminal 2: start the browser interface:

```bash
source ~/.venvs/hacktoberfest-2026/bin/activate
cd /path/to/hacktoberfest-2026/dev-challenge/weekend-challenge
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
```

## Use the Browser Demo

Once Streamlit reports that the app is running, keep both WSL terminals open
and follow these steps. Review the AI-generated text against the photos before
confirming or exporting it.

1. Open [http://localhost:8501](http://localhost:8501) in your Windows browser.
2. Choose a demo sample or upload a reference photo and a current photo.
3. Click **Analyze both photos**.
4. Review and correct both observations against the photos, then confirm them.
5. Click **Draft cleanup checklist**.
6. Review and edit the final checklist against the photos, then confirm it.
7. Click **Download reviewed result** if you want to preserve the experiment as JSON.
