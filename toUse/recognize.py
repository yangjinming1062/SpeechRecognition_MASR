import _init_path
import platform
from models.conv import GatedConv

system_type = platform.system()
if(system_type == 'Windows'):
    model = GatedConv.load("SpeechRecognition_MASR\\pretrained\\gated-conv.pth")
    #import scipy
    #_,receipt_data = scipy.io.wavfile.read("E:\\打开欢呼比.wav")
    #text = model.predict(receipt_data)事实证明效果相同
    text = model.predict("E:\\打开欢呼比.wav")
elif(system_type == 'Linux'):
    model = GatedConv.load('SpeechRecognition_MASR/pretrained/gated-conv.pth')
    text = model.predict("/media/yangjinming/DATA/Dataset/PrimeWords/d/d2/d25104a2-6be0-4950-9ec0-42e8e1303492.wav")

print("识别结果:",text)
