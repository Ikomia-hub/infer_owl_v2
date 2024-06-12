from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_owl_v2.infer_owl_v2_process import InferOwlV2Param
from torch.cuda import is_available

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferOwlV2Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferOwlV2Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        # Models name
        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model")
        self.combo_model.addItem("google/owlv2-base-patch16-ensemble")
        self.combo_model.addItem("google/owlv2-base-patch16")
        self.combo_model.addItem("google/owlv2-base-patch16-finetuned")
        self.combo_model.addItem("google/owlv2-large-patch14")
        self.combo_model.addItem("google/owlv2-large-patch14-finetuned")
        self.combo_model.setCurrentText(self.parameters.model_name)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(
            self.grid_layout, "Prompt", self.parameters.prompt)

        # Confidence thresholds
        self.spin_conf_thres_box = pyqtutils.append_double_spin(
                                                self.grid_layout,
                                                "Confidence threshold boxes",
                                                self.parameters.conf_thres,
                                                min=0., max=1.,
                                                step=0.01, decimals=2)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.parameters.cuda
        self.parameters.conf_thres = self.spin_conf_thres_box.value()
        self.parameters.prompt = self.edit_prompt.text()
        # self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)

# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferOwlV2WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_owl_v2"

    def create(self, param):
        # Create widget object
        return InferOwlV2Widget(param, None)
