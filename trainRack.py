


class Rack:

    def __init__(self):
        self.dataset = None
        self.set_basic_menu()
        self.set_device()

        pass

    def set_model(self,model):
        pass

    def set_basic_menu(self):
        pass

    def set_device(self):
        pass

    def set_optimizer(self,optimizer_class):
        pass

    def set_criterion(selfself,criterion):
        pass

    def add_arguments(self,arg_dict):
        pass

    def parse_args(self):
        pass

    def prepare_output(self):
        pass

    def get_loaders(self):
        return (None, None)

    def launch(self):
        pass
        self.prepare_output()

        self.train_loader,self.test_loader = self.get_loaders()

        self.train()

    def train(self):
        pass

    def test(self):
        pass