flow_img_wh = (832, 256)
content_img_wh = (128, 128)
valid_tasks = ["content", "depth", "flow"]
valid_enc_Ds = {"ResNet50": 2048, "ResNet18": 512, "ViT-B": 768, "ConvNeXt": 768}
valid_encs = valid_enc_Ds.keys()
