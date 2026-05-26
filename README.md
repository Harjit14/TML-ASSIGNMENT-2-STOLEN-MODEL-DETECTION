# TML Assignment 2 - Stolen Model Detection

This project contains the code used to score suspect models for the stolen model detection task and submit the generated CSV file.

## Files

- `task_template.py` downloads the model files from Hugging Face, loads the target and suspect ResNet-18 models, computes similarity-based scores, and writes `submission.csv`.
- `submission_file.py` uploads `submission.csv` to the task submission API.
- `requirements.txt` lists the Python packages needed to run both scripts.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Update the placeholder values:

- In `task_template.py`, replace `Your Hugging face token` with your Hugging Face token.
- In `task_template.py`, replace `Repo id` with the Hugging Face model repository ID.
- In `submission_file.py`, replace `api-id` with your actual API key.

## Generate Submission

Run:

```bash
python task_template.py
```

This downloads the model files, evaluates all 360 suspect models, and creates:

```text
submission.csv
```


## Submit

After `submission.csv` is created and `API_KEY` is configured, run:

```bash
python submission_file.py
```

And the final output prints in the leaderboard
