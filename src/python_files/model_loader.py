from transformers import TFT5ForConditionalGeneration
from os import listdir


class ModelLoader:
    """
    Model loader to load different checkpoints of models trained on the TPU
    """
    def __init__(self, model_size, root_folder, model_folder) -> None:
        """
        Init
        :param model_size: Model size
        :type model_size: str
        :param root_folder: Root folder of project
        :type root_folder: str
        :param model_folder: folder of model
        :type model_folder: str
        """
        super().__init__()
        self.model = TFT5ForConditionalGeneration.from_pretrained(model_size)
        self.root_folder = root_folder
        self.model_folder = model_folder

    def get_list_of_models(self):
        """
        Get List of different checkpoints
        :return: List of checkpoints
        :rtype: list
        """
        list_model_folder = listdir(self.root_folder + "/" + self.model_folder)
        list_model_folder = [self.root_folder + "/" + self.model_folder + "/" + item for item in list_model_folder]

        n_epochs = (len(list_model_folder) - 2) / 2
        model_dict = {}
        for file_path in list_model_folder:
            file_path_x = file_path.split(".")
            if file_path_x[-1] == "index":
                file_name = ".".join(file_path_x[:-1])
                # get the epch as index example: t5_cnn_daily_mail-{epoch}.ckpt
                model_dict[int(".".join(file_path_x[:-2]).split("-")[-1])] = file_name

        return [item for key, item in sorted(model_dict.items())]

    def yield_models(self):
        """
        Iterable get model function
        :yield model
        """
        model_list = self.get_list_of_models()
        #         print(model_list)
        for model_file in model_list:
            print(model_file)

            self.model.load_weights(model_file)
            yield self.model

    def load_epoch(self, epoch):
        """
        Load special epoch
        :param epoch: Epoch
        :type epoch: int
        :return: Model
        :rtype: model
        """
        model_list = self.get_list_of_models()
        self.model.load_weights(model_list[epoch])
        return self.model