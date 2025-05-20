# Visual Misinformation in Vision-Language Models

This repository contains the code and dataset for the paper *"I'll believe it when I see it: Images increase misinformation sharing in Vision-Language Models"*.

The project investigates how the presence of images influences misinformation resharing behavior in vision-language models (VLMs). It includes simulation tools for persona conditioning, prompt construction, model interfacing, and statistical analysis.


## 📁 Project Structure

- `src/`: Main source code.
  - `main_exec.py`: Entry point for running simulations.
  - `load_cnfg.py`: Loads experiment configurations and parameters.
  - `complete.py`, `models.py`: Interfaces and wrappers for VLMs.
  - `prompt.py`: Constructs prompts for input to VLMs.
  - `conversation.py`: Manages dialogue flow and response collection.
  - `crawl.py`: Scrapes news articles from PolitiFact.
  - `classify_img.py`, `classify_news.py`, `clean_data.py`: Preprocess and classify news data and associated images.
  - `save_res.py`, `scan_res.py`: Save and aggregate experimental results.
  - `infstat.py`, `plot.py`: Statistical analysis and plotting utilities.

- `data/`: Input data.
  - `dialogs_user.json`: Prompt templates using third-person framing.
  - `dialogs_asst.json`: Prompt templates using second-person framing.
  - `demo_small.json`: Demographic attribute definitions.
  - `news_200.json`: Text content of the news dataset.
  - `trait.json`: Trait keyword definitions for persona prompts.
  - `.key.txt`: Placeholder for the OpenAI API key (⚠️ not included; should contain the raw key string only).
  - `.anth.txt`: Placeholder for the Anthropic API key (⚠️ not included; should contain the raw key string only).
  - `.hf.txt`: Placeholder for the Hugging Face API token (⚠️ not included; should contain the raw key string only).


-   `imgs/`: News-related images used in the dataset (provided empty). Available at: [Download 200-News Dataset (Google Drive)](https://drive.google.com/drive/folders/1U3HPyt4NktwLExbcwWQZNLCH4uMwiyW_)


-   `res/`: Stores the results generated from simulation runs (provided empty).

-   `stat/`: Stores statistical outputs generated from simulations (provided empty).



## ⚙️ Requirements
This project uses `Python 3.12.3`. You can install dependencies via:

```
$ pip install -r requirements.txt
```

**Note:** `crawl.py` was originally developed with `numpy==1.12.1`, while the rest of the codebase requires `numpy==1.26.4`. Since `crawl.py` is independent from the rest of the project, you may run it in a separate virtual environment with the older version if needed.


## 📦 Dataset

The dataset of 200 news items is available at the following link:

🔗 [Download 200-News Dataset (Google Drive)](https://drive.google.com/drive/folders/1U3HPyt4NktwLExbcwWQZNLCH4uMwiyW_)

After downloading, place the `imgs/` folder at the same level as the `src/` and `data/` directories.

The corresponding text data (`news_200.json`) is already included in the `data/` directory.

**Disclaimer:** This dataset contains material (such as text and images) that may be protected by copyright and owned by third parties. We do not claim any rights over such content. All copyrights remain with their respective owners. This dataset is shared solely for non-commercial research and educational purposes.



## 🚀 Running the Code

To view available command-line arguments, use the `-h` flag:

```
$ python3 main_exec.py -h
```

Use the `-v` option to visualize simulation progress across news items.

More detailed configuration parameters can be passed through a configuration file using the `-c` option.

For example, `cnfg_gpt.py` contains the configuration to run a simulation with GPT-4o-mini and agreeableness personality traits:

```
$ python3 main_exec.py -c cnfg_gpt -v
```

Another example, `cnfg_cld.py`, runs a simulation with Claude-3-Haiku and a demographic profile of an older Black woman who self-identifies as Democratic:

```
$ python3 main_exec.py -c cnfg_cld -v
```


## 📄 License

Released under the MIT License.


