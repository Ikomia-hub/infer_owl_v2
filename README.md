<div align="center">
  <img src="images/logo.png" alt="Algorithm icon">
  <h1 align="center">infer_owl_v2</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_owl_v2">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_owl_v2">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_owl_v2/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_owl_v2.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>


Run OWLv2 a zero-shot text-conditioned object detection model.
![OWL](https://raw.githubusercontent.com/Ikomia-hub/infer_owl_v2/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display


# Init your workflow
wf = Workflow()    

# Add the OWLv2 Object Detector
owl = wf.add_task(name="infer_owl_v2", auto_connect=True)

# Run on your image  
# wf.run_on(path="path/to/your/image.png")
wf.run_on(url='http://images.cocodataset.org/val2017/000000039769.jpg')

# Inspect your results
display(owl.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'google/owlv2-base-patch16-ensemble':  The OWLv2 algorithm has different checkpoint models,
    - google/owlv2-base-patch16-ensemble"
    - google/owlv2-base-patch16"
    - google/owlv2-base-patch16-finetuned"
    - google/owlv2-large-patch14"
    - google/owlv2-large-patch14-finetuned"
- **prompt** (str) - default 'a cat, remote control': Text prompt for the model
- **conf_thres** (float) - default '0.2': Box threshold for the prediction‍
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display


# Init your workflow
wf = Workflow()    

# Add the OWLv2 Object Detector
owl = wf.add_task(name="infer_owl_v2", auto_connect=True)
owl.set_parameters({
    "model_name":"google/owlv2-base-patch16-ensemble",
    "prompt":"a cat, remote control",
    "conf_thres":"0.25",
    "cuda":"True"
})

# Run on your image  
# wf.run_on(path="path/to/your/image.png")
wf.run_on(url='http://images.cocodataset.org/val2017/000000039769.jpg')

# Inspect your results
display(owl.get_image_with_graphics())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_owl_v2", auto_connect=True)

# Run on your image  
wf.run_on(url='http://images.cocodataset.org/val2017/000000039769.jpg')

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```