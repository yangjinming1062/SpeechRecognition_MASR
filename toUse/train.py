import _init_path
from models.conv import GatedConv

model = GatedConv.load("SpeechRecognition_MASR/pretrained/gated-conv.pth")
model.to_train()
model.fit("data/train.index", "data/dev.index",train_batch_size=2)
