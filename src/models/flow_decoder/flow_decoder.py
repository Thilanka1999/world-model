from torch import nn
from .pwcnet import PWCDecoder
from .mask_retriever import MaskRetriever


class FlowDecoder(nn.Module):
    """
    Performs flow retreival (PWC-Net), and then mask retreival
    """

    def __init__(self, enc_dims) -> None:
        super().__init__()
        self.decoder = PWCDecoder(enc_dims)
        self.mask_retiever = MaskRetriever()

    def forward(self, feature_list_1, feature_list_2, img_hw):
        optical_flows = self.decoder(feature_list_1, feature_list_2, img_hw)
        optical_flows_rev = self.decoder(feature_list_2, feature_list_1, img_hw)
        masks = self.mask_retiever(optical_flows, optical_flows_rev)

        img1_valid_masks = masks["img1_valid_masks"]
        img2_valid_masks = masks["img2_valid_masks"]
        fwd_flow_diff_pyramid = masks["fwd_flow_diff_pyramid"]
        bwd_flow_diff_pyramid = masks["bwd_flow_diff_pyramid"]

        info = {  # in TrianFlow, only the zeroth index of each of the following are returned since they calculate the loss within. But we need all for our independent FlowLoss
            "optical_flows_rev": optical_flows_rev,
            "optical_flows": optical_flows,
            "img1_valid_masks": img1_valid_masks,
            "img2_valid_masks": img2_valid_masks,
            "fwd_flow_diff_pyramid": fwd_flow_diff_pyramid,
            "bwd_flow_diff_pyramid": bwd_flow_diff_pyramid,
            "flow_pred": optical_flows[0],
        }

        return info
