import copy
from ikomia import core, dataprocess, utils
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
import os
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import Image
import numpy as np

# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferOwlV2Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.prompt = "a cat, remote control"
        self.model_name = "google/owlv2-base-patch16-ensemble"
        self.conf_thres = 0.2
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = params["model_name"]
        self.cuda = utils.strtobool(params["cuda"])
        self.conf_thres = float(params["conf_thres"])
        self.prompt = params["prompt"]
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {}
        params["model_name"] = str(self.model_name)
        params["prompt"] = str(self.prompt)
        params["conf_thres"] = str(self.conf_thres)
        params["cuda"] = str(self.cuda)
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferOwlV2(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Example :  self.add_input(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferOwlV2Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        self.model = None
        self.model_name = None
        self.processor = None
        self.device = torch.device("cpu")
        torch.set_grad_enabled(False)

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) \
                            + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def load_model(self):
        param = self.get_param_object()
        self.device = torch.device(
                "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        self.model_name = param.model_name
        try:
            self.processor = Owlv2Processor.from_pretrained(
                                                param.model_name,
                                                cache_dir=self.model_folder,
                                                local_files_only=True
                                            )
        except Exception as e:
            self.processor = Owlv2Processor.from_pretrained(
                                                param.model_name,
                                                cache_dir=self.model_folder,
                                            )

        try:
            self.model = Owlv2ForObjectDetection.from_pretrained(
                                                    param.model_name,
                                                    cache_dir=self.model_folder,
                                                    local_files_only=True
                                                ).to(self.device)
        except Exception as e:
            self.model = Owlv2ForObjectDetection.from_pretrained(
                                                param.model_name,
                                                cache_dir=self.model_folder,
                                                ).to(self.device)

        param.update = False

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        if self.model is None:
            if self.model_name != param.model_name:
                self.load_model()

        # Process input
        texts = [param.prompt.split(",")]
        inputs = self.processor(text=texts, images=src_image, return_tensors="pt").to(self.device)

        # Inference
        outputs = self.model(**inputs)

        # Determine the width and height ratios
        w, h = src_image.shape[:2]
        if w > h:
            width_ratio = 1
            height_ratio = h / w
        else:
            width_ratio = w / h
            height_ratio = 1

        # Convert outputs (bounding boxes and class logits) to COCO API
        unnormalized_image = self.get_preprocessed_image(inputs.pixel_values.squeeze().cpu())
        # unnormalized_image = self.get_preprocessed_image(inputs.pixel_values)
        w_un_img, h_un_img=  unnormalized_image.size[::-1]
        unnormalized_image = unnormalized_image.crop((0, 0, w_un_img, h * h_un_img/w))

        # Resize it to the original size
        unnormalized_image = unnormalized_image.resize((w,h))

        # Get results
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        results = self.processor.post_process_object_detection(
                                    outputs=outputs,
                                    target_sizes=target_sizes,
                                    threshold=param.conf_thres
                                )

        # Retrieve predictions corresponding text queries
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        # set class name
        classe_names = texts[0]
        self.set_names(classe_names)

        # Add object output
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            cls = int(label.item())
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = tuple(box)
            x1 = x1 / width_ratio
            y1 = y1 / height_ratio
            x2 = x2 / width_ratio
            y2 = y2 / height_ratio
            w = float(x2 - x1)
            h = float(y2 - y1)
            self.add_object(i, cls, float(score), float(x1), float(y1), w, h)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferOwlV2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_owl_v2"
        self.info.short_description = "Run OWLv2 a zero-shot text-conditioned object detection model"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/logo.png"
        self.info.authors = "Minderer, M., Gritsenko, A., & Houlsby, N."
        self.info.article = "Scaling Open-Vocabulary Object Detection"
        self.info.journal = "NeurIPS"
        self.info.year = 2023
        self.info.license = "Apache-2.0"
        # URL of documentation
        self.info.documentation_link = "https://proceedings.neurips.cc/paper_files/paper/2023/file/e6d58fc68c0f3c36ae6e0e64478a69c0-Paper-Conference.pdf"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_owl_v2"
        self.info.original_repository = "https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit"
        # Python version
        self.info.min_python_version = "3.10.0"
        # Keywords used for search
        self.info.keywords = "Zero-shot, CLIP, ViT, PyTorch"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return InferOwlV2(self.info.name, param)
