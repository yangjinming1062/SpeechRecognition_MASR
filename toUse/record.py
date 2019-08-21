import os
import requests
import threading
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import wave

import _init_path
from models.conv import GatedConv
import feature


class FileRecord():
    def __init__(self,CHUNK=2000,RATE=16000):
        self.filename = None
        self.allowRecording = False
        self.CHUNK = CHUNK
        self.RATE = RATE
        self.ani = SubplotAnimation(fun_use=True)
        self.wav_list=[]
        self.label_list=[]
        self.intUI()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.mainloop()


    def intUI(self):
        self.root = tkinter.Tk()
        self.root.title('wav音频录制')
        x = (self.root.winfo_screenwidth()-200)//2
        y = (self.root.winfo_screenheight()-140)//2
        self.root.geometry('430x200+{}+{}'.format(x,y))
        self.root.resizable(False,False)
        self.btStart = tkinter.Button(self.root,text='开始录音',command=self.start)
        self.btStart.place(x=50,y=20,width=100,height=40)
        self.btStop = tkinter.Button(self.root,text='停止录音',command=self.stop)
        self.btStop.place(x=50,y=80,width=100,height=40)
        self.btShowWav = tkinter.Button(self.root,text='Real-Time Wav',command=self.ShowWav)
        self.btShowWav.place(x=180,y=20,width=200,height=40)
        self.btTrain = tkinter.Button(self.root,text='开始训练',command=self.real_time_train)
        self.btTrain.place(x=180,y=80,width=100,height=40)
        self.btSumTrain = tkinter.Button(self.root,text='总体训练',command=self.sum_train)
        self.btSumTrain.place(x=280,y=80,width=100,height=40)
        self.label = tkinter.Text()
        self.label.place(x=50,y=140,width=330,height=40)
    

    def real_time_train(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE,
                            input=True, frames_per_buffer=self.CHUNK)
        data = []
        while True:
            y = np.frombuffer(stream.read(self.CHUNK), dtype=np.int16)
            data.append(y)
            if self.ani._valid(np.array(data[-24::1]).flatten()):
                if len(data)>3:
                    label = self.label.get('0.0', 'end').replace('\n','')
                    wav = np.array(data).flatten()
                    #todo
                    self.wav_list.append(wav)
                    self.label_list.append(label)
                    break
                data=data[-2:]
        stream.stop_stream()
        stream.close()
        p.terminate()
    

    def sum_train(self):
        #train.train_am(self.wav_list,self.label_list)
        pass


    def start(self):
        self.filename = tkinter.filedialog.asksaveasfilename(filetypes=[('Sound File','*.wav')])
        if not self.filename:
            return
        if not self.filename.endswith('.wav'):
            self.filename = self.filename+'.wav'
        self.allowRecording = True
        self.root.title('正在录音...')
        threading.Thread(target=self.record).start()


    def stop(self):
        self.allowRecording = False
        self.root.title('wav音频录制')


    def ShowWav(self):
        self.ani = SubplotAnimation()
        plt.show()

    
    def close(self):
        if self.allowRecording:
            tkinter.messagebox.showerror('正在录音','请先停止录音')
            return
        self.root.destroy()


    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format = pyaudio.paInt16,channels=1,rate = self.RATE,
                        input = True,frames_per_buffer=self.CHUNK)
        wf = wave.open(self.filename,'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.RATE)
        while self.allowRecording:#从录音设备读取数据，直接写入wav文件
            data = stream.read(self.CHUNK)
            wf.writeframes(data)
        wf.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.filename = None



class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, path = None,serviceAddress='http://172.16.100.29:5000/recognize/',fun_use=False):
        self.httpService = serviceAddress
        #音频波形动态显示，实时显示波形，实时进行离散傅里叶变换分析频域
        if path is not None and os.path.isfile(path):
            self.stream = wave.open(path)
            self.rate = self.stream.getparams()[2]
            self.chunk = int(self.rate/1000*125)
            self.read = self.stream.readframes
        else:
            self.rate = 16000
            self.chunk = 2000
            p = pyaudio.PyAudio()
            self.stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate,
                            input=True, frames_per_buffer=self.chunk)
            self.read = self.stream.read
        self.yysb = GatedConv.load("E:\\AboutPython\\AboutDL\\语音识别MASR\\pretrained\\gated-conv.pth")

        self.data = []

        #fig = plt.figure(num='Real-time wave')
        fig,ax = plt.subplots()

        self.t = np.linspace(0, self.chunk - 1, self.chunk)
        self.line1, = ax.plot([], [], lw=2)
        ax.set_xlim(0, self.chunk)
        ax.set_ylim(-5000, 5000)

        interval = int(1000*self.chunk/self.rate)#更新间隔/ms
        if not fun_use:
            animation.TimedAnimation.__init__(self, fig, interval=interval, blit=True)


    def _valid(self,check_wav):
        '''
        判断是否开始、停止记录声音的方法，返回布尔结果
        if处可能需要根据情况设计更好的判断条件
        当返回为True时，开始、停止记录声音，False则记录声音
        '''
        check = np.array([abs(x) for x in check_wav]).sum()/len(check_wav)
        #if check_wav.max()<900 and check_wav.min()>-900:#未听到声音
        if check > 80:
            return True
        else:
            return False


    def _draw_frame(self, framedata):
        x = np.linspace(0, self.chunk - 1, self.chunk)
        y = np.frombuffer(self.read(self.chunk), dtype=np.int16)
        special_flag = False#特殊判断标记，当最后一段音频不足时赋值为真，主要就是针对读取固定长度音频的情况
        if len(y) == 0:
            return
        if len(y)<self.chunk:
            y = np.pad(y,(0,self.chunk-len(y)),'constant')#数据维度需要和坐标维度一致
            special_flag = True
        self.data.append(y)
        if special_flag or self._valid(np.array(self.data[-16::1]).flatten()):
            #修改语音识别调用方式：这种是在开始记录有效声音后直到准备清理数据时最后用完整数据调用一次
            if len(self.data)>5:
                wav = np.array(self.data).flatten()
                if True:#本地方式
                    han = self.yysb.predict(wav)
                else:#发送到服务器的方式
                    try:
                        han = requests.post(self.httpService, json={'token':'SR', 'data':wav,'pre_type':'H'})
                        han.encoding='utf-8'
                        han = han.text
                    except BaseException as e:
                        han = str(e)
                print('识别汉字：{}'.format(han))#todo:或者给需要的地方

            self.data=self.data[-4:]
        self.data = self.data[-16::1]
        # 音频图
        self.line1.set_data(x, y)
        self._drawn_artists = [self.line1]


    def new_frame_seq(self):
        return iter(range(self.t.size))


    def _init_draw(self):
        self.line1.set_data([], [])


if __name__ == "__main__":
    ani = SubplotAnimation()
    plt.show()
    #rec = FileRecord()