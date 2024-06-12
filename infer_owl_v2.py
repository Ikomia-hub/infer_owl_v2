from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_owl_v2.infer_owl_v2_process import InferOwlV2Factory
        return InferOwlV2Factory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_owl_v2.infer_owl_v2_widget import InferOwlV2WidgetFactory
        return InferOwlV2WidgetFactory()
