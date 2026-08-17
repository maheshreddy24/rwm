
<h1 align="center">
IntPhys 2
</h1>
<h3 align="center">
<a href="https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip">Dataset</a> &nbsp; | &nbsp;
 <a href="https://huggingface.co/datasets/facebook/IntPhys2">Hugging Face</a> &nbsp; | &nbsp;
 <a href="https://arxiv.org/abs/2506.09849">Paper</a> &nbsp; | &nbsp;
 <a href="https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks">Blog</a>
</h3>

![IntPhys2 intro image](https://github.com/facebookresearch/IntPhys2/blob/main/IntPhys2_github.png "IntPhys2 benchmark")

IntPhys 2 is a video benchmark designed to evaluate the intuitive physics understanding of deep learning models. Building on the original [IntPhys benchmark](https://intphys.cognitive-ml.fr/), IntPhys 2 focuses on four core principles related to  macroscopic objects: Permanence, Immutability, Spatio-Temporal Continuity, and Solidity. These conditions are inspired by research into intuitive physical understanding emerging during early childhood. IntPhys 2 offers a comprehensive suite of tests, based on the violation of expectation framework, that challenge models to differentiate between possible and impossible events within controlled and diverse virtual environments. Alongside the benchmark, we provide performance evaluations of several state-of-the-art models. Our findings indicate that while these models demonstrate basic visual understanding, they face significant challenges in grasping intuitive physics across the four principles in complex scenes, with most models performing at chance levels (50\%), in stark contrast to human performance, which achieves near-perfect accuracy. This underscores the gap between current models and human-like intuitive physics understanding, highlighting the need for advancements in model architectures and training methodologies.

This codebase contains:

- download links for IntPhys2
- dataloaders
- code to evaluate MLLMs and prediction based models on IntPhys2
- list of the unreal engine assets and plugins we bought to create IntPhys2 in [Unreal Engine 5.4](https://www.unrealengine.com/en-US/blog/unreal-engine-5-4-is-now-available).
  
**IntPhys2 benchmark splits**
=====================================

We release three separate splits. The first is intended for debugging only and provide some measurement on the model's sensitivity to the video generation artifacts (such as mp4 compression or cloud moving the background of the scene). The second is the main evaluation set with three different sub-splits ("Easy", "Medium", "Hard"). The third is a held-out split that we release without additional metadata.

| Split        | Scenes | Videos | Description                                                                                   | Purpose              |
|--------------|--------|--------|-----------------------------------------------------------------------------------------------|----------------------|
| Debug Set    | 5      | 60     | Static cameras, bright assets, 3 generations                                                 | Model calibration   |
| Main Set     | 253    | 1,012  | Static and moving cameras: 3 sub-splits:<br>- Easy: Simple environments, colorful shapes<br>- Medium: Diverse backgrounds, textured shapes<br>- Hard: Realistic objects, complex backgrounds | Main evaluation set  |
| Held-Out Set | 86     | 344    | Moving cameras, Mirrors hard sub-split, includes distractors                                  | Test set        |


## Downloading the benchmark
IntPhys2 is available on [Hugging Face](https://huggingface.co/datasets/facebook/IntPhys2) or by [direct download](https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip
).

## Evaluating on the Held-Out set
We are not releasing the metadata associated with the held-out set to prevent training data contamination, we invite researchers to upload the results in the following [Leaderboard](https://huggingface.co/spaces/facebook/physical_reasoning_leaderboard). The model_answer column in the resulting jsonl file should contain either 1 if the video is deemed possible by the model or 0 if it's not possible. 

The order does not matter since what matter is to have the correct video id in row_id, here is an example of what a line of your jsonl file should looks like:
`{"data_name": "intphys2", "task": "HeldOut", "row_id": "ed97b1c631746a17ba76af61df949cc9ac5fbaa14bc8f3e1afac1f8cf73d5078", "model_answer": 0} `

So your row_id will just map directly to the ground truth annotation file.

## Evaluation code for MLLMS
We provide the code to run evauation with MLLMs in different files. To run open sources model like Qwen-VL 2.5 uisng the hugging face transformers library, you can use the file `IntPhys2_transformers.py`. You just need to specify the dataset path in the variable INTPHYS2_DATA_FOLDER. To run OpenAI models from the official API, you can leverage the file `IntPhys2_openai.py` in which you need to specify your API keys in YOUR_API_KEY, YOUR_API_ENDPOINT variable. Lastly to run Gemini models from the google API, you can use `IntPhys2_google_api.py` and update the YOUR_API_KEY variable.

The output will be stored in the `Results` folder.

## Evaluation code for prediction based models

We provide the code to run prediction based evaluations in the `prediction_evals` subfolder

### Running the code

For algorithmic clarity and reproducibility, we provide a version of our code which can be used to extract surprise metrics from models. It is compatible with V-JEPA models and VideoMAEv2. The code is based on [github.com/facebookresearch/jepa-intuitive-physics](https://github.com/facebookresearch/jepa-intuitive-physics).

For requirements to run the code, see `requirements.txt` .

We provide a singular evaluations:
- `intphys2` This evaluation will run through the dataset and extract surprises for all models. These surprises can then be used to compute accuracy.

To run the evaluation code, the file `evaluation_code/evals/intphys2/utils.py` 

As the code is meant to be reusable on various clusters where data doesn't share a common path. You need to specify what is `CLUSTER` as well as what the paths of the datasets are.
If you intend on only using a singular cluster, the `get_cluster()` function can simply be replaced by:
```python
@lru_cache()
def get_cluster() -> str:
    return CLUSTER
```
Then, just update the dataset paths in `DATASET_PATHS_BY_CLUSTER`.

From the `evaluation_code` folder, evaluations can either be run locally, e.g:
```bash
python -m evals.main --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 --fname evals/intphys2/configs/vjepa_rope.yaml
```

or through submitit, e.g.:

```bash
python -m evals.main_distributed --fname evals/intphys2/configs/vjepa_rope.yaml --folder ./logs --partition PARTITION 
```

### Configurations

We provide default configurations in the evaluations folder that should be adapted depending on the model that you are using.

The *model_kwargs* section contains information to load the pretrained model. Most important are *checkpoint* which is the model path, and *module_name* which is the wrapper to use.

The parameters *tasks_per_node* and *nodes* are only used when using submitit to control the number of GPUs used. Due to the computational cost of COSMOS, we recomment running on 8 nodes with 8 task per nodes each. Other models can be run on 1 node.

## License

IntPhys 2 is licensed under the CC BY-NC 4.0 license.  Third party content pulled from other locations are subject to their own licenses and you may have other legal obligations or restrictions that govern your use of that content.
The use of IntPhys 2 is limited to evaluation purposes, where it can be utilized to generate tags for classifying visual content, such as videos and images. All other uses, including generative AI applications that create or automate new content (e.g. audio, visual, or text-based), are prohibited.

## Citing IntPhys2
If you use IntPhys2, please cite:
```
@misc{bordes2025intphys2benchmarkingintuitive,
      title={IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments}, 
      author={Florian Bordes and Quentin Garrido and Justine T Kao and Adina Williams and Michael Rabbat and Emmanuel Dupoux},
      year={2025},
      eprint={2506.09849},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.09849}, 
}
```
