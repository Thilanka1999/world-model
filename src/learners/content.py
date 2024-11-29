import torch
from ..models.backbone import BackBone
from ..util import make_obj_from_conf
from typing import Dict, Sequence, Any
from omegaconf.dictconfig import DictConfig


class ContentDecoder(torch.nn.Module):
    def __init__(self, input_size=768, hidden_size=8192, output_size=8192):
        super(ContentDecoder, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(
            input_size, hidden_size
        )  # Input layer to hidden layer
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.relu = torch.nn.ReLU()  # ReLU activation function
        self.fc2 = torch.nn.Linear(
            hidden_size, hidden_size
        )  # Hidden layer to hidden layer
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.fc3 = torch.nn.Linear(
            hidden_size, output_size
        )  # Hidden layer to output layer

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ContentLearner(torch.nn.Module):
    device_count = 2

    def __init__(self, encoder: BackBone | Dict[str, Any] | DictConfig) -> None:
        super().__init__()

        if type(encoder) in [dict, DictConfig]:
            encoder = make_obj_from_conf(encoder)
        else:
            assert isinstance(encoder, torch.nn.Module), type(encoder)

        self.encoder = encoder
        self.decoder = ContentDecoder(input_size=encoder.dims["l6"])

    def set_devices(self, devices: Sequence[int]) -> None:
        self.devices = devices
        self.encoder.cuda(devices[0])
        self.decoder.cuda(devices[1])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        view1, view2 = batch["view1"], batch["view2"]
        view1, view2 = view1.cuda(self.devices[0]), view2.cuda(self.devices[0])
        out1 = self.encoder(view1)
        out2 = self.encoder(view2)
        X_one, X_two = out1["emb"], out2["emb"]
        X_one, X_two = X_one.cuda(self.devices[1]), X_two.cuda(self.devices[1])
        Y_one = self.decoder(X_one)
        Y_two = self.decoder(X_two)

        return {
            "X_one": X_one,
            "X_two": X_two,
            "Y_one": Y_one,
            "Y_two": Y_two,
        }
