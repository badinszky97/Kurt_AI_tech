import random

class DataKeeper:
    def __init__(self):
        self.RawDatas = []
        self.RawLabel = []
    def AddNewType(self, data, label):
        self.RawDatas.append(data)
        self.RawLabel.append(label)
    
    @property
    def KeptTypeCount(self):
        return len(self.RawDatas)
    
    def GetData(self, index):
        return self.RawDatas[index]
    
    def GenerateTrainTestValid(self, alpha=0.8):
        #ha alpha 0.8 -> train: 80%, valid: 10%, test: 10%
        SumRawDatas = []
        #létrehozzuk a teljes adatállományt egy tömbben
        for x in range(0, len(self.RawDatas)):
            for element in self.RawDatas[x]:
                SumRawDatas.append([element,self.RawLabel[x]])
        #összekeverjük
        random.shuffle(SumRawDatas)
        #szétválogatjuk 3 részre
        train = SumRawDatas[:int(len(SumRawDatas)*alpha)]
        test_valid = SumRawDatas[int(len(SumRawDatas)*alpha):]
        test = test_valid[:int(len(test_valid)*0.5)]
        valid = test_valid[int(len(test_valid)*0.5):]
        
        #x_train, y_train létrehozása
        self.x_train = []
        self.y_train = []
        for x in train:
            self.x_train.append(x[0])
            self.y_train.append(x[1])
            
        #x_test, y_test létrehozása
        self.x_test = []
        self.y_test = []
        for x in test:
            self.x_test.append(x[0])
            self.y_test.append(x[1])
            
        #x_valid, y_valid létrehozása
        self.x_valid = []
        self.y_valid = []
        for x in train:
            self.x_valid.append(x[0])
            self.y_valid.append(x[1])
            
        #numpy tömbbe konvertálás
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)
        
        self.x_valid = np.array(self.x_valid)
        self.y_valid = np.array(self.y_valid) 
        print("Kész")
        
        
def WavToSplittedArray(filename="", window_size = 100):
    fs, wave = scipy.io.wavfile.read(filename) 
    windows_count = int(len(wave)/window_size)

    kimenet = []
    for i in range(windows_count):
      buffer = []
      for x in range(window_size):
        buffer.append(wave[i*window_size+x])
      kimenet.append(buffer)
    return kimenet