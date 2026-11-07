# TRACE

TRACE is an undergraduate thesis project that studies how to detect backdoor activation in large language models (LLMs).

A backdoor can make a model behave normally until a specific word, phrase, or condition changes its behavior. TRACE looks at a prompt, the model's answer, and information from the generation process. It then gives a score that indicates possible backdoor activation.

**TRACE is a research prototype. A warning is a prediction, and a normal result does not prove that the model is backdoor-free.**

The full thesis title is **TRACE: A Lightweight Dual-View Framework for Backdoor Activation Detection in Selected Open Source Large Language Models**. The authors are Gil Jose F. Penaflor and Neil David L. Villanueva, BS Computer Science students at Ateneo de Naga University.

## Which branch should I read?

This is **`main`**, which contains the earlier TRACE prototype and its original Llama 2 experiment.

The [`trace-update-2026-11-07` branch](https://github.com/giljosepenaflor/TRACE/tree/trace-update-2026-11-07) contains the newer research workspace, expanded training and testing, external baseline comparisons, and later experiments. Read that branch for the November 7 development snapshot. Its result tables describe different detector versions and test sets.

## How it works

TRACE uses two views of the same interaction:

| View | What it looks at |
| --- | --- |
| Behavioral view | The meaning of the prompt and the answer, represented as numbers by a separate sentence encoder. |
| Generation view | How uncertain or confident the target model was while generating each token, which is a word or part of a word. |

The generation view summarizes three values: average token entropy, average generated-token confidence, and confidence variance. These describe uncertainty, the probability assigned to the tokens actually generated, and how much that confidence changes across the answer.

Full TRACE combines both views and passes them to a small classifier. We also train Behavioral-only and Generation-only versions to check whether combining the views helps.

```text
Prompt -> target LLM -> answer and token statistics
                              |
                    TRACE extracts both views
                              |
                    Classifier score and status
```

The behavioral view uses MiniLM, a separate text encoder. It does not read Llama 2's internal hidden-layer states. The original TRACE implementation trains a small projection of the text embeddings and a logistic-regression classifier. The target LLM stays unchanged during detection.

## What you can do

- Try the local demonstration with synthetic examples.
- Use the configured remote Llama 2 model and see a TRACE assessment beside each answer.
- Open **View analysis** to inspect scores, generation features, and model details.
- Export an interaction as JSON.
- Use experimental Chat mode for conversation memory. Its detection scores have not been validated for conversations.

## Try the synthetic demo

This is the easiest way to see the interface. It uses made-up examples and does not run an LLM or test a real backdoor.

Open PowerShell in the project folder. For a new setup, create the environment and install the core packages:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then start the demo:

```powershell
.\.venv\Scripts\python.exe -m traceguard demo
.\.venv\Scripts\python.exe -m traceguard serve
```

Open **http://127.0.0.1:8765**. Select an interaction and click **Analyze interaction**. Keep the terminal open while using the page. Press Ctrl+C to stop it.

## Use real Llama 2 inference

The real setup uses `meta-llama/Llama-2-7b-chat-hf` with the `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` adapter. The GPU server runs the model. Your computer runs the dashboard and TRACE detector.

This mode needs the model packages, trusted detector files, the MiniLM encoder files, and access to the configured GPU server. **Cloning this repository alone does not include all of these files.**

To install the local model packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

On a computer that already has the runtime files and credentials configured:

1. Start the Vast GPU instance and its model service.
2. Open the SSH tunnel to that service.
3. Run the local dashboard:

```powershell
.\start-vast.ps1
```

Open **http://127.0.0.1:8769**. Select **Independent prompt**, enter a harmless question, and send it. Open the risk card or **View analysis** beside the answer for more details.

The tunnel and model service must stay running. An API or detector error is shown as an error or an unavailable score. The app does not replace a failed request with a synthetic answer.

See the [live dashboard guide](docs/LLAMA2_DASHBOARD.md) and [Vast setup guide](docs/VAST_MIGRATION.md) for connection details. Those guides include older entries; check the dated updates and current instance details. The [Colab guide](docs/COLAB_SETUP.md) covers the alternative notebook setup.

## Results included on this branch

The original Llama 2 experiment evaluated the frozen detector on 30 reviewed question groups. Each group includes clean/backdoored and trigger absent/present responses. Ten of the 40 collected question groups were excluded because their reviews were unresolved or disagreed.

The table below uses the 60 interactions from the backdoored model. The 60 clean-reference interactions were checked separately. There were only five verified activations in the seen-trigger split and seven in the unseen-trigger split.

| Method | Seen-trigger F1 | Unseen-trigger F1 |
| --- | ---: | ---: |
| Full TRACE | 0.667 | 0.750 |
| Behavioral-only | 0.625 | 0.750 |
| Generation-only | 0.476 | 0.571 |

F1 balances how often flagged examples are correct with how many actual activations are found. Higher is better; it is not the same as accuracy. Here, "unseen" means the trigger was excluded from detector training and validation. Researchers had already examined it in a separate pilot.

Full TRACE scored higher than Generation-only in this small experiment, but tied Behavioral-only on unseen-trigger F1. The small sample and wide uncertainty intervals do not establish a reliable advantage from combining the views. See the [original held-out report](docs/MTBA_HELDOUT_RESULTS.md) for the controls, exclusions, and full results.

These are the earlier results. The [November 7 branch](https://github.com/giljosepenaflor/TRACE/tree/trace-update-2026-11-07) contains a larger, different evaluation. Do not combine its numbers with this table or treat results from different test sets as a direct improvement comparison.

This branch also preserves earlier synthetic and TinyLlama engineering work. Those examples are separate from the Llama 2 research results. The prepared three-source expansion on this branch is an earlier workflow; completed expanded experiments are on the November 7 branch.

## What the results mean

A triggered prompt is not automatically an activated backdoor. We review the actual answers and compare the backdoored model with its clean base, both with and without the trigger. Uncertain cases stay out of supervised training and evaluation.

Training teaches the classifier from labeled examples. Validation is used to choose settings and the decision threshold. Held-out testing checks the frozen detector on separate examples. A higher validation score does not guarantee a higher test score.

The live score is a classifier output, not a calibrated probability. For example, a score of 0.8 does not establish an 80% chance of activation. Live interactions keep unknown ground truth and are kept out of training and final research evaluation.

TRACE does not remove backdoors. Our completed activation-detection experiments are limited to the evaluated Llama 2 configuration. They do not establish reliable detection across other models or all backdoor types.

## Project files

| Location | What it contains |
| --- | --- |
| `traceguard/` | Feature extraction, detector code, and the local interface. |
| `scripts/` | Collection, review, training, and evaluation workflows. |
| `remote/` | The API used by the remote GPU model service. |
| `tests/` | Checks for feature calculations, data separation, and application behavior. |
| `docs/` | Setup guides, experiment plans, and result reports. |
| `docs/thesis/` | The preserved thesis draft and working project context. |

## Run the software checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Some checks need local research files, cached models, or an optional model environment. These may not be available in a fresh clone. Software checks verify the implementation; they do not prove detection accuracy.

## Research notes

Use the [project context](docs/thesis/PROJECT_CONTEXT.md) for the research scope and the [development status](docs/DEVELOPMENT_STATUS.md) for the detailed work record. The [interaction format](docs/INTERACTION_FORMAT.md) explains the saved data fields. The [earlier README notes](docs/README_HISTORY.md) preserve the old project updates.

Private research responses, review workbooks, trained artifacts, model caches, and credentials are kept outside this source repository. The live setup and research workflows need additional local files.
